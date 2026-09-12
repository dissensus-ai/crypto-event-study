"""
C8c: Mechanical sweep under the CANONICAL (rolling) winsorisation.
=================================================================

The published Section 4.8 selection-bias sweep (no-filter 0.58x, relaxed 1.49x,
2-asset 1.61x, strict 1.31x) was estimated on returns winsorised with a GLOBAL
0.5%/99.5% clip (see c2_relaxed_threshold_sensitivity.load_returns_panel), while
the curated 5.7x headline used the project's CANONICAL rolling winsorisation
(data_preparation.winsorize_returns: 30-day rolling mean +/- 5 sigma). That
inconsistency confounds the 5.7x -> 1.6x curated-vs-mechanical attribution: part
of the gap could be the winsorisation rule, not the event-selection rule.

This script re-runs the mechanical sweep on the 135-candidate census, holding the
candidate MEMBERSHIP fixed (the census stage2_*_pass flags), but estimating the
GJR-GARCH-X on returns winsorised with the canonical rolling rule. It reports the
multiplier + Welch p at each threshold side-by-side with:
  - the published global-clip numbers, and
  - a freshly recomputed global-clip baseline (same code path, global clip), so
    the comparison is strictly like-for-like under one estimator.

Specs (census flags):
  nofilter  -> stage2_nofilter_pass (135)   [published 0.58x]
  relaxed   -> stage2_relaxed_pass  (115)   [published 1.49x]
  twoasset  -> stage2_std_pass      ( 93)   [published 1.61x]
  strict    -> stage2_strict_pass   ( 77)   [published 1.31x]

Also re-estimates the curated S1 50-event sample under BOTH winsorisations so the
like-for-like curated-vs-mechanical gap is reported under each rule.

Uses FastTARCHX for the point deltas (many fits); validated identical to canonical
to ~3 dp.

Outputs:
    results/c8c-mechanical-winsor-per-asset.csv
    results/c8c-mechanical-winsor-summary.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2  # noqa: E402
from tarch_x_fast import FastTARCHX  # noqa: E402

# canonical rolling winsoriser
PKG_CODE = c2.CODE_DIR
sys.path.insert(0, str(PKG_CODE))
from data_preparation import DataPreparation  # noqa: E402

OUT_DIR = c2.OUT_DIR
DATA_DIR = c2.DATA_DIR
ASSETS = c2.ASSETS

PUBLISHED = {"nofilter": 0.58, "relaxed": 1.49, "twoasset": 1.61, "strict": 1.31}
SPEC_FLAG = {
    "nofilter": "stage2_nofilter_pass",
    "relaxed": "stage2_relaxed_pass",
    "twoasset": "stage2_std_pass",
    "strict": "stage2_strict_pass",
}


def load_returns_rolling():
    """Returns panel winsorised with the CANONICAL rolling rule (30d, 5 sigma)."""
    dp = DataPreparation(data_path=str(DATA_DIR))
    panel = {}
    for a in ASSETS:
        df = pd.read_csv(DATA_DIR / f"{a}.csv")
        df["date"] = pd.to_datetime(df["snapped_at"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        df = df.loc[c2.START_DATE:c2.END_DATE]
        logret = np.log(df["price"]).diff() * 100
        ret_w = dp.winsorize_returns(logret.dropna(), window=30, n_std=5.0)
        panel[a] = ret_w.dropna()
    return panel


def fit_deltas(returns, exog, seed=0):
    fast = FastTARCHX(returns.values, exog.values)
    p, f, ok = fast.fit_multistart(n_starts=5, seed=seed)
    d = fast.deltas(p)
    return d[0], d[1], ok


def get_spec_dates(spec, events, census):
    if spec == "curated":
        return c2.get_event_dates_for_spec("S1_baseline", events, census)
    flag = SPEC_FLAG[spec]
    sub = census.loc[census[flag]]
    inf = sub.loc[sub["tentative_category"] == "Infrastructure", "date"].tolist()
    reg = sub.loc[sub["tentative_category"] == "Regulatory", "date"].tolist()
    return inf, reg


def run_panel(panel, sent, events, census, spec, seed=0):
    inf_d, reg_d = get_spec_dates(spec, events, census)
    di_list, dr_list, rows = [], [], []
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
        dum = c2.build_event_dummies(r.index, inf_d, reg_d,
                                     c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s = sent.reindex(r.index).fillna(0)
        exog = pd.concat([dum[["D_infrastructure", "D_regulatory"]],
                          s[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
                         axis=1).fillna(0)
        di, dr, ok = fit_deltas(r, exog, seed=seed)
        di_list.append(di); dr_list.append(dr)
        rows.append({"spec": spec, "asset": a, "dInfra": di, "dReg": dr, "converged": ok,
                     "n_infra": len(inf_d), "n_reg": len(reg_d)})
    infra = np.array(di_list); reg = np.array(dr_list)
    mult = np.nanmean(infra) / np.nanmean(reg) if np.nanmean(reg) != 0 else np.nan
    t, p = stats.ttest_ind(infra, reg, equal_var=False, nan_policy="omit")
    return {"spec": spec, "n_infra": len(inf_d), "n_reg": len(reg_d),
            "mean_dInfra": np.nanmean(infra), "mean_dReg": np.nanmean(reg),
            "multiplier": mult, "welch_t": t, "welch_p": p}, rows


def main():
    print("Loading sentiment/events/census...")
    panel_global = c2.load_returns_panel()          # global 0.5/99.5 clip
    panel_rolling = load_returns_rolling()          # canonical rolling 30d/5sigma
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel_global.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])

    specs = ["curated", "nofilter", "relaxed", "twoasset", "strict"]
    summary, per_asset = [], []

    for winlabel, pnl in [("global_clip", panel_global), ("rolling_canon", panel_rolling)]:
        print(f"\n--- winsorisation: {winlabel} ---")
        for spec in specs:
            srow, rows = run_panel(pnl, sent, events, census, spec, seed=0)
            srow["winsor"] = winlabel
            srow["published_global_mult"] = PUBLISHED.get(spec, np.nan)
            summary.append(srow)
            for rr in rows:
                rr["winsor"] = winlabel
                per_asset.append(rr)
            print(f"  {spec:9s} n=({srow['n_infra']},{srow['n_reg']}) "
                  f"infra={srow['mean_dInfra']:.3f} reg={srow['mean_dReg']:.3f} "
                  f"mult={srow['multiplier']:.2f}x p={srow['welch_p']:.3f} "
                  f"[pub {PUBLISHED.get(spec, float('nan'))}]")

    df_s = pd.DataFrame(summary)
    df_s.to_csv(OUT_DIR / "c8c-mechanical-winsor-summary.csv", index=False)
    pd.DataFrame(per_asset).to_csv(OUT_DIR / "c8c-mechanical-winsor-per-asset.csv", index=False)
    print("\nSaved: c8c-mechanical-winsor-summary.csv, c8c-mechanical-winsor-per-asset.csv")


if __name__ == "__main__":
    main()
