#!/usr/bin/env python3
"""
GATE: Returns (first-moment) event study on the UNIFIED variance basis.
======================================================================

Purpose
-------
The no-structure (returns) paper estimates its headline null on a 4-asset
(BTC/ETH/SOL/ADA), negative-valence-only subset (8 infra vs 7 reg events) and
finds CAR_infra=-7.6% vs CAR_reg=-11.1%, diff=+3.6pp, p~=0.81 (block bootstrap).

The companion variance (infra/event-study) paper is estimated on a DIFFERENT
basis: 6 assets (BTC/ETH/XRP/BNB/LTC/ADA) and 50 events (binary infra/reg).

For the merged paper, BOTH moments must be reported on ONE common basis. We
adopt the variance basis and re-run the SAME returns methodology on it. This
script reuses the EXACT engine (ConstantMeanModel CAR, event-equal-weighted
block bootstrap and event-level Welch t-test) and changes only the
sample.

It does THREE things:
  1. SMOKE TEST  -- reproduce the published headline on the ORIGINAL sample
                    (4 assets from the Binance parquet cache, Infra_Negative /
                    Reg_Negative events from events_reclassified.json).
  2. GATE (A)    -- 6-asset basis (incl. XRP), 50 events from events.csv.
  3. GATE (B)    -- 5-asset basis (ex-XRP), 50 events from events.csv.

This is ANALYSIS ONLY. It writes NEW files only:
  - c-gate-returns-unified-results.csv  (one row per basis)
  - c11-returns-results.csv             (byte-identical release alias)
  - prints a FINDING summary to stdout
It touches nothing existing.

Methodology faithfully matched to:
  - code/src/event_study.py            (ConstantMeanModel, window (-5,30),
                                         inclusive 250-calendar-day span,
                                         estimation ending on event day -31)
  - code/scripts/run_corrected_bootstrap.py (event-equal-weighted block
                                         bootstrap: average across assets
                                         within event FIRST, resample whole
                                         events, seed 42, 5000 reps)
  - code/scripts/run_im_test.py        (Welch t-test on event-level mean CARs)
"""

import sys
import os
from itertools import combinations
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.event_study import ConstantMeanModel, MarketModel

# ----------------------------------------------------------------------------
# Fixed methodology parameters (mirror config / scripts exactly)
# ----------------------------------------------------------------------------
WINDOW = (-5, 30)
COMPLETE_EVENT_WINDOW_N = WINDOW[1] - WINDOW[0] + 1  # inclusive: 36 days
N_BOOTSTRAP = 5000
SEED = config.RANDOM_SEED          # 42
ESTIMATION_WINDOW = config.ESTIMATION_WINDOW_DAYS   # 250
GAP_WINDOW = config.GAP_WINDOW_DAYS                 # 30

DATA = CODE_DIR / 'data'
OUT_DIR = Path(os.environ.get('CES_OUT_DIR', CODE_DIR.parent / 'results'))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# CAR engine (identical to ConstantMeanModel.compute_abnormal_returns)
# ----------------------------------------------------------------------------
MODEL = ConstantMeanModel(estimation_window=ESTIMATION_WINDOW, gap_window=GAP_WINDOW)


def event_level_cars(
    returns_dict,
    events,
    window=WINDOW,
    require_complete_event_window=False,
    return_coverage=False,
):
    """For each event: list of {event_id, mean_car, asset_cars}.

    mean_car = average of per-asset CARs (within-event averaging FIRST), which
    is the equal-event-weighting scheme used by run_corrected_bootstrap.py and
    run_im_test.py. Events with zero valid assets are dropped (engine returns
    'error' when an asset has no data / insufficient estimation window).

    The historical smoke test retains the predecessor engine's permissive
    event-window rule.  The unified gate sets ``require_complete_event_window``
    and therefore admits an asset-event cell only when all 36 calendar-day
    returns in [-5,+30] are present.  This check is deliberately separate from
    the legacy estimation lookback, whose inclusive calendar slicing is left
    unchanged (251 observations when the full 250-day span is available).
    """
    out = []
    coverage = {
        'asset_event_cells_total': len(events) * len(returns_dict),
        'asset_event_cells_usable': 0,
        'asset_event_cells_missing': 0,
        'missing_estimation_cells': 0,
        'incomplete_event_window_cells': 0,
        'other_error_cells': 0,
        'missing_asset_event_ids': [],
    }
    for ev in events:
        date = ev['date']
        asset_cars = {}
        for sym, ret in returns_dict.items():
            res = MODEL.compute_abnormal_returns(ret, date, window)
            missing_reason = None
            if 'error' in res:
                if str(res['error']).startswith('Insufficient estimation data'):
                    missing_reason = 'estimation'
                elif str(res['error']).startswith('Insufficient event window data'):
                    missing_reason = 'event_window'
                else:
                    missing_reason = 'other'
            elif (require_complete_event_window and
                  res.get('event_window_n') != COMPLETE_EVENT_WINDOW_N):
                missing_reason = 'event_window'

            if missing_reason is None:
                asset_cars[sym] = res['car']
                coverage['asset_event_cells_usable'] += 1
            else:
                coverage['asset_event_cells_missing'] += 1
                coverage['missing_asset_event_ids'].append(
                    f"{int(ev['event_id'])}:{sym}"
                )
                if missing_reason == 'estimation':
                    coverage['missing_estimation_cells'] += 1
                elif missing_reason == 'event_window':
                    coverage['incomplete_event_window_cells'] += 1
                else:
                    coverage['other_error_cells'] += 1
        if asset_cars:
            out.append({
                'event_id': ev['event_id'],
                'date': date,
                'mean_car': float(np.mean(list(asset_cars.values()))),
                'n_assets': len(asset_cars),
            })
    assert (coverage['asset_event_cells_usable'] +
            coverage['asset_event_cells_missing'] ==
            coverage['asset_event_cells_total'])
    coverage['missing_asset_event_ids'] = ';'.join(
        coverage['missing_asset_event_ids']
    )
    if return_coverage:
        return out, coverage
    return out


def block_bootstrap_diff(infra_means, reg_means, n_boot=N_BOOTSTRAP, seed=SEED):
    """Event-level block bootstrap of the difference in mean CARs.

    Mirrors CorrectedEventBlockBootstrap.bootstrap_difference_test: resample
    whole events (the event-level mean CARs) with replacement within each
    group independently, recompute group means, take difference. Two-tailed
    p against zero, percentile 95% CI.
    """
    rng = np.random.default_rng(seed)
    infra_means = np.asarray(infra_means, dtype=float)
    reg_means = np.asarray(reg_means, dtype=float)
    na, nb = len(infra_means), len(reg_means)
    orig_diff = infra_means.mean() - reg_means.mean()

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(infra_means, size=na, replace=True)
        b = rng.choice(reg_means, size=nb, replace=True)
        diffs[i] = a.mean() - b.mean()

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    if orig_diff >= 0:
        p_two = 2 * np.mean(diffs <= 0)
    else:
        p_two = 2 * np.mean(diffs >= 0)
    p_two = float(min(p_two, 1.0))
    # one-sided p in the direction of the point estimate
    p_one = float(min(p_two / 2.0, 1.0))
    return {
        'diff': float(orig_diff),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_two': p_two,
        'p_one': p_one,
        'se': float(diffs.std()),
    }


def exact_permutation_diff(infra_means, reg_means):
    """Enumerate every event-label assignment and test the absolute mean gap.

    This is feasible for the predecessor negative-only subset: assigning eight
    of 15 event-level CARs to infrastructure gives C(15,8)=6,435 assignments.
    Ties are included in the two-sided tail, including the observed assignment.
    """
    infra_means = np.asarray(infra_means, dtype=float)
    reg_means = np.asarray(reg_means, dtype=float)
    pooled = np.concatenate([infra_means, reg_means])
    n_infra = len(infra_means)
    observed = float(infra_means.mean() - reg_means.mean())
    threshold = abs(observed)
    extreme = 0
    assignments = 0
    all_idx = np.arange(len(pooled))
    for infra_idx_tuple in combinations(range(len(pooled)), n_infra):
        infra_idx = np.fromiter(infra_idx_tuple, dtype=int, count=n_infra)
        is_infra = np.zeros(len(pooled), dtype=bool)
        is_infra[infra_idx] = True
        diff = pooled[is_infra].mean() - pooled[all_idx[~is_infra]].mean()
        extreme += int(abs(diff) >= threshold)
        assignments += 1
    return {
        'diff': observed,
        'assignments': assignments,
        'extreme_assignments': extreme,
        'p_two': float(extreme / assignments),
    }


def market_model_event_level_cars(returns_dict, events, market_proxy='BTC'):
    """Event-equal CARs from the packaged BTC-proxy ``MarketModel``.

    The proxy's return series is supplied once.  The packaged implementation
    then fits alpha and beta separately for every non-BTC asset/event cell and
    uses its constant-mean fallback for BTC itself.  This robustness calculation
    belongs to the predecessor four-asset negative-only smoke sample, so it
    retains that sample's historical permissive event-window rule.
    """
    model = MarketModel(
        estimation_window=ESTIMATION_WINDOW,
        gap_window=GAP_WINDOW,
        market_proxy=market_proxy,
    )
    model.set_market_returns(returns_dict[market_proxy])
    out = []
    for ev in events:
        asset_cars = {}
        for sym, ret in returns_dict.items():
            res = model.compute_abnormal_returns(
                ret,
                ev['date'],
                WINDOW,
                symbol=sym,
            )
            if 'error' not in res:
                asset_cars[sym] = res['car']
        if asset_cars:
            out.append({
                'event_id': ev['event_id'],
                'date': ev['date'],
                'mean_car': float(np.mean(list(asset_cars.values()))),
                'n_assets': len(asset_cars),
            })
    return out


def im_test(infra_means, reg_means):
    """Ordinary Welch t-test on event-level means.

    The historical `im_*` output-column names are retained for file
    compatibility; this is not the Ibragimov-Mueller partition procedure.
    """
    g1 = np.asarray(infra_means, dtype=float)
    g2 = np.asarray(reg_means, dtype=float)
    n1, n2 = len(g1), len(g2)
    m1, m2 = g1.mean(), g2.mean()
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    tcrit = stats.t.ppf(0.975, df)
    return {
        'mean_infra': float(m1),
        'mean_reg': float(m2),
        'diff': float(m1 - m2),
        'se': float(se),
        't': float(t),
        'df': float(df),
        'p': float(p),
        'ci_low': float((m1 - m2) - tcrit * se),
        'ci_high': float((m1 - m2) + tcrit * se),
    }


def pooled_obs_bootstrap_diff(infra_obs, reg_obs, n_boot=N_BOOTSTRAP, seed=SEED):
    """OBSERVATION-weighted bootstrap of the difference (the ORIGINAL scheme
    behind the published -7.6/-11.1/p=0.81 headline). Pools all per-asset CARs,
    resamples observations with replacement. Used in the smoke test only, to
    show both published numbers reconcile."""
    rng = np.random.default_rng(seed)
    a = np.asarray(infra_obs, dtype=float)
    b = np.asarray(reg_obs, dtype=float)
    orig = a.mean() - b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, size=len(a), replace=True).mean() - \
                   rng.choice(b, size=len(b), replace=True).mean()
    if orig >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    return {
        'mean_infra': float(a.mean()),
        'mean_reg': float(b.mean()),
        'diff': float(orig),
        'ci_low': float(np.percentile(diffs, 2.5)),
        'ci_high': float(np.percentile(diffs, 97.5)),
        'p_two': float(min(p, 1.0)),
    }


def mde_note(infra_means, reg_means, alpha=0.05, power=0.80):
    """Minimum detectable effect (two-sample t, equal-ish n) + Cohen's d of the
    observed effect, matching the paper's power discussion."""
    g1 = np.asarray(infra_means, float)
    g2 = np.asarray(reg_means, float)
    n1, n2 = len(g1), len(g2)
    sp = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2))
    d_obs = (g1.mean() - g2.mean()) / sp if sp > 0 else np.nan
    # MDE in Cohen's d for a two-sample test with harmonic-mean n per group
    n_h = 2.0 / (1.0 / n1 + 1.0 / n2)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    d_mde = (z_a + z_b) * np.sqrt(2.0 / n_h)
    mde_pp = d_mde * sp  # back to CAR units (decimal CAR over window)
    return {
        'pooled_sd': float(sp),
        'cohens_d_obs': float(d_obs),
        'cohens_d_mde': float(d_mde),
        'mde_car': float(mde_pp),  # in CAR decimal units (e.g. 0.40 = 40pp)
    }


# ============================================================================
# SMOKE TEST -- original sample (4 assets from cache, reclassified events)
# ============================================================================
def load_cache_returns(symbols):
    """Load returns from the Binance parquet cache (original engine's source)."""
    import glob
    rd = {}
    for sym in symbols:
        # full-range cache file: SYM_ohlcv_2019-01-01_2026-01-29.parquet
        matches = sorted(glob.glob(str(DATA / 'cache' / f'{sym}_ohlcv_2019-01-01_*.parquet')))
        if not matches:
            print(f"  [smoke] {sym}: NO full-range cache file -> skip")
            continue
        df = pd.read_parquet(matches[-1])
        if 'returns' in df.columns:
            rd[sym] = df['returns'].dropna()
            print(f"  [smoke] {sym}: {len(rd[sym])} returns (cache)")
    return rd


def load_reclassified_neg_events():
    import json
    d = json.load(open(DATA / 'events_reclassified.json'))
    by_type = {}
    for e in d['events']:
        if not e.get('include_in_reanalysis', True):
            continue
        if not e.get('meets_impact_threshold', False):
            continue
        if not e.get('has_sufficient_estimation_data', True):
            continue
        et = e.get('type_detailed', e['type'])
        by_type.setdefault(et, []).append(e)
    return by_type


def run_smoke():
    print("\n" + "=" * 74)
    print("SMOKE TEST -- reproduce published headline on ORIGINAL sample")
    print("  (4 assets BTC/ETH/SOL/ADA from cache; Infra_Negative/Reg_Negative)")
    print("=" * 74)

    symbols = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]   # BTC,ETH,SOL,ADA
    rd = load_cache_returns(symbols)
    by_type = load_reclassified_neg_events()
    infra_ev = by_type.get('Infra_Negative', [])
    reg_ev = by_type.get('Reg_Negative', [])
    print(f"  Infra_Negative events: {len(infra_ev)} | Reg_Negative events: {len(reg_ev)}")

    infra_el = event_level_cars(rd, infra_ev)
    reg_el = event_level_cars(rd, reg_ev)
    infra_means = [e['mean_car'] for e in infra_el]
    reg_means = [e['mean_car'] for e in reg_el]
    print(f"  Events with valid CARs: infra={len(infra_means)}, reg={len(reg_means)}")

    # event-equal-weighted block bootstrap (the 'corrected' / IM-consistent run)
    bb = block_bootstrap_diff(infra_means, reg_means)
    im = im_test(infra_means, reg_means)
    perm = exact_permutation_diff(infra_means, reg_means)

    # Packaged BTC-proxy market-model robustness on the same negative subset.
    mm_infra_el = market_model_event_level_cars(rd, infra_ev, market_proxy='BTC')
    mm_reg_el = market_model_event_level_cars(rd, reg_ev, market_proxy='BTC')
    mm_infra_means = [e['mean_car'] for e in mm_infra_el]
    mm_reg_means = [e['mean_car'] for e in mm_reg_el]
    mm_bb = block_bootstrap_diff(mm_infra_means, mm_reg_means)
    mm_im = im_test(mm_infra_means, mm_reg_means)
    mm_perm = exact_permutation_diff(mm_infra_means, mm_reg_means)

    # observation-weighted pooled bootstrap (the ORIGINAL -7.6/-11.1/0.81 headline)
    infra_obs, reg_obs = [], []
    for ev in infra_ev:
        for sym, ret in rd.items():
            r = MODEL.compute_abnormal_returns(ret, ev['date'], WINDOW)
            if 'error' not in r:
                infra_obs.append(r['car'])
    for ev in reg_ev:
        for sym, ret in rd.items():
            r = MODEL.compute_abnormal_returns(ret, ev['date'], WINDOW)
            if 'error' not in r:
                reg_obs.append(r['car'])
    obs = pooled_obs_bootstrap_diff(infra_obs, reg_obs)

    print("\n  --- OBSERVATION-WEIGHTED pooled bootstrap (PUBLISHED HEADLINE) ---")
    print(f"    CAR_infra = {obs['mean_infra']*100:+.1f}%   CAR_reg = {obs['mean_reg']*100:+.1f}%")
    print(f"    diff = {obs['diff']*100:+.1f}pp   p(two)={obs['p_two']:.3f}   "
          f"CI=[{obs['ci_low']*100:+.1f}%, {obs['ci_high']*100:+.1f}%]")
    print(f"    [paper says: -7.6% / -11.1% / +3.6pp / p=0.81 / CI [-25.3,+30.9]]")

    print("\n  --- EVENT-EQUAL-WEIGHTED block bootstrap (paper robustness) ---")
    print(f"    CAR_infra = {im['mean_infra']*100:+.1f}%   CAR_reg = {im['mean_reg']*100:+.1f}%")
    print(f"    diff = {bb['diff']*100:+.1f}pp   p(two)={bb['p_two']:.3f}   "
          f"CI=[{bb['ci_low']*100:+.1f}%, {bb['ci_high']*100:+.1f}%]")
    print(f"    [paper says: -7.9% / -9.4% / +1.5pp / p=0.93]")

    print("\n  --- EVENT-LEVEL WELCH test (paper robustness) ---")
    print(f"    diff = {im['diff']*100:+.1f}pp   t={im['t']:.2f}   df={im['df']:.1f}   "
          f"p={im['p']:.3f}   CI=[{im['ci_low']*100:+.1f}%, {im['ci_high']*100:+.1f}%]")
    print(f"    [paper says: +1.5pp / t=0.09 / p=0.93 / CI [-32.5,+35.4]]")

    print("\n  --- EXACT EVENT-LABEL PERMUTATION (negative subset) ---")
    print(f"    assignments={perm['assignments']}  extreme={perm['extreme_assignments']}  "
          f"p(two)={perm['p_two']:.6f}")

    print("\n  --- BTC-PROXY MARKET MODEL (packaged MarketModel; negative subset) ---")
    print(f"    valid events: infra={len(mm_infra_means)}, reg={len(mm_reg_means)}")
    print(f"    CAR_infra={mm_im['mean_infra']*100:+.6f}%  "
          f"CAR_reg={mm_im['mean_reg']*100:+.6f}%  "
          f"diff={mm_im['diff']*100:+.6f}pp")
    print(f"    block p(two)={mm_bb['p_two']:.6f}  Welch p={mm_im['p']:.6f}  "
          f"exact permutation p={mm_perm['p_two']:.6f}")

    return {
        'obs': obs, 'bb': bb, 'im': im,
        'perm': perm,
        'market_model_bb': mm_bb,
        'market_model_im': mm_im,
        'market_model_perm': mm_perm,
        'n_infra': len(infra_means), 'n_reg': len(reg_means),
        'market_model_n_infra': len(mm_infra_means),
        'market_model_n_reg': len(mm_reg_means),
    }


# ============================================================================
# GATE RUN -- unified variance basis (50 events, 6 or 5 assets, CoinGecko CSV)
# ============================================================================
def load_csv_returns(symbols):
    """Load returns from the committed CoinGecko price CSVs (shared with the
    variance paper; byte-identical across both repos)."""
    rd = {}
    for sym in symbols:
        df = pd.read_csv(DATA / f'{sym.lower()}.csv')
        df['date'] = pd.to_datetime(df['snapped_at'].str.replace(' UTC', '', regex=False))
        df = df.sort_values('date').set_index('date')
        ret = df['price'].pct_change().dropna()
        rd[sym] = ret
    return rd


def load_unified_events():
    ev = pd.read_csv(DATA / 'events.csv')
    infra = ev[ev['type'] == 'Infrastructure'].to_dict('records')
    reg = ev[ev['type'] == 'Regulatory'].to_dict('records')
    return infra, reg


def run_gate(label, symbols):
    print("\n" + "=" * 74)
    print(f"GATE BASIS [{label}] -- assets: {symbols}")
    print("  50 events from events.csv (binary Infrastructure/Regulatory)")
    print("=" * 74)

    rd = load_csv_returns(symbols)
    for s in symbols:
        print(f"  {s}: {len(rd[s])} returns ({rd[s].index.min().date()} -> {rd[s].index.max().date()})")
    infra_ev, reg_ev = load_unified_events()
    print(f"  events.csv: {len(infra_ev)} infra, {len(reg_ev)} reg")

    infra_el, infra_coverage = event_level_cars(
        rd,
        infra_ev,
        require_complete_event_window=True,
        return_coverage=True,
    )
    reg_el, reg_coverage = event_level_cars(
        rd,
        reg_ev,
        require_complete_event_window=True,
        return_coverage=True,
    )
    infra_means = [e['mean_car'] for e in infra_el]
    reg_means = [e['mean_car'] for e in reg_el]
    print(f"  events with valid CARs: infra={len(infra_means)}, reg={len(reg_means)}")

    coverage = {}
    for key in (
        'asset_event_cells_total',
        'asset_event_cells_usable',
        'asset_event_cells_missing',
        'missing_estimation_cells',
        'incomplete_event_window_cells',
        'other_error_cells',
    ):
        coverage[key] = infra_coverage[key] + reg_coverage[key]
    missing_ids = []
    for ids in (
        infra_coverage['missing_asset_event_ids'],
        reg_coverage['missing_asset_event_ids'],
    ):
        if ids:
            missing_ids.extend(ids.split(';'))
    missing_ids.sort(key=lambda cell: (int(cell.split(':', 1)[0]), cell))
    coverage['missing_asset_event_ids'] = ';'.join(missing_ids)
    print(
        "  asset-event coverage: "
        f"{coverage['asset_event_cells_usable']}/"
        f"{coverage['asset_event_cells_total']} usable; "
        f"{coverage['asset_event_cells_missing']} missing "
        f"({coverage['missing_estimation_cells']} estimation, "
        f"{coverage['incomplete_event_window_cells']} incomplete event window)"
    )
    print(f"  omitted cells: {coverage['missing_asset_event_ids'] or 'none'}")

    bb = block_bootstrap_diff(infra_means, reg_means)
    im = im_test(infra_means, reg_means)
    mde = mde_note(infra_means, reg_means)

    print(f"\n  CAR_infra = {im['mean_infra']*100:+.2f}%   CAR_reg = {im['mean_reg']*100:+.2f}%")
    print(f"  diff (infra - reg) = {bb['diff']*100:+.2f}pp")
    print(f"  block-bootstrap p: two-sided={bb['p_two']:.3f}  one-sided={bb['p_one']:.3f}")
    print(f"  block-bootstrap 95% CI: [{bb['ci_low']*100:+.2f}%, {bb['ci_high']*100:+.2f}%]")
    print(f"  Event-level Welch: t={im['t']:.2f}  df={im['df']:.1f}  p={im['p']:.3f}  "
          f"CI=[{im['ci_low']*100:+.2f}%, {im['ci_high']*100:+.2f}%]")
    print(f"  Cohen's d (observed) = {mde['cohens_d_obs']:+.3f}   "
          f"pooled SD = {mde['pooled_sd']*100:.1f}pp")
    print(f"  MDE @ 80% power, a=.05: d={mde['cohens_d_mde']:.2f}  "
          f"=> {mde['mde_car']*100:.1f}pp (much larger than observed effect)")

    return {
        'basis': label,
        'assets': '/'.join(symbols),
        'n_infra': len(infra_means),
        'n_reg': len(reg_means),
        **coverage,
        'car_infra_pct': im['mean_infra'] * 100,
        'car_reg_pct': im['mean_reg'] * 100,
        'diff_pp': bb['diff'] * 100,
        'block_p_two': bb['p_two'],
        'block_p_one': bb['p_one'],
        'block_ci_low_pct': bb['ci_low'] * 100,
        'block_ci_high_pct': bb['ci_high'] * 100,
        'im_t': im['t'],
        'im_df': im['df'],
        'im_p': im['p'],
        'im_ci_low_pct': im['ci_low'] * 100,
        'im_ci_high_pct': im['ci_high'] * 100,
        'cohens_d_obs': mde['cohens_d_obs'],
        'pooled_sd_pp': mde['pooled_sd'] * 100,
        'mde_car_pp': mde['mde_car'] * 100,
    }


def main():
    print("#" * 74)
    print("# GATE: returns event study on the UNIFIED variance basis")
    print(f"# window={WINDOW}  est={ESTIMATION_WINDOW}d  gap={GAP_WINDOW}d  "
          f"boot={N_BOOTSTRAP}  seed={SEED}")
    print("#" * 74)

    smoke = run_smoke()
    rowA = run_gate("A: 6-asset (incl XRP)", ['BTC', 'ETH', 'XRP', 'BNB', 'LTC', 'ADA'])
    rowB = run_gate("B: 5-asset (ex XRP)", ['BTC', 'ETH', 'BNB', 'LTC', 'ADA'])

    # The predecessor-sample robustness fields are repeated on both unified
    # basis rows so the two-row release schema remains backward compatible.
    legacy_fields = {
        'legacy_negative_assets': 'BTC/ETH/SOL/ADA',
        'legacy_negative_n_infra': smoke['n_infra'],
        'legacy_negative_n_reg': smoke['n_reg'],
        'legacy_negative_constant_car_infra_pct': smoke['im']['mean_infra'] * 100,
        'legacy_negative_constant_car_reg_pct': smoke['im']['mean_reg'] * 100,
        'legacy_negative_constant_diff_pp': smoke['im']['diff'] * 100,
        'legacy_negative_constant_block_p_two': smoke['bb']['p_two'],
        'legacy_negative_constant_welch_p': smoke['im']['p'],
        'legacy_negative_exact_permutation_assignments': smoke['perm']['assignments'],
        'legacy_negative_exact_permutation_extreme': smoke['perm']['extreme_assignments'],
        'legacy_negative_exact_permutation_p_two': smoke['perm']['p_two'],
        'legacy_negative_market_proxy': 'BTC',
        'legacy_negative_market_n_infra': smoke['market_model_n_infra'],
        'legacy_negative_market_n_reg': smoke['market_model_n_reg'],
        'legacy_negative_market_car_infra_pct': smoke['market_model_im']['mean_infra'] * 100,
        'legacy_negative_market_car_reg_pct': smoke['market_model_im']['mean_reg'] * 100,
        'legacy_negative_market_diff_pp': smoke['market_model_im']['diff'] * 100,
        'legacy_negative_market_block_p_two': smoke['market_model_bb']['p_two'],
        'legacy_negative_market_welch_p': smoke['market_model_im']['p'],
        'legacy_negative_market_permutation_p_two': smoke['market_model_perm']['p_two'],
    }
    rowA.update(legacy_fields)
    rowB.update(legacy_fields)

    out = pd.DataFrame([rowA, rowB])
    out_path = OUT_DIR / 'c-gate-returns-unified-results.csv'
    alias_path = OUT_DIR / 'c11-returns-results.csv'
    out.to_csv(out_path, index=False)
    out.to_csv(alias_path, index=False)
    print("\n" + "=" * 74)
    print(f"Saved results CSV: {out_path}")
    print(f"Saved release alias: {alias_path}")
    print("=" * 74)

    print("\nGATE VERDICT")
    print("-" * 74)
    for r in (rowA, rowB):
        sig = "first-moment non-rejection (p>0.10)" if r['block_p_two'] > 0.10 \
              else "first-moment result approaches or crosses the chosen threshold"
        print(f"  [{r['basis']}] diff={r['diff_pp']:+.2f}pp  "
              f"block-p={r['block_p_two']:.3f}  Welch-p={r['im_p']:.3f}  => {sig}")
    print("-" * 74)


if __name__ == '__main__':
    main()
