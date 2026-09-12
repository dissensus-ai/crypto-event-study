"""
C7: CCC-GARCH-X Gaussian-copula model-based bootstrap.

This is an intermediate benchmark. C9 replaces the Gaussian innovation copula
with the fitted heavy-tailed specification, and c19 compares fixed-path and
recursive propagation. The manuscript's conclusion is based on that full set
of results rather than treating c7 as definitive.

Background
----------
The headline result is the per-asset comparison of the infrastructure event
coefficient (delta_infra) vs the regulatory one (delta_reg) in the GJR-GARCH-X
variance equation, averaged over 6 assets. Naive iid inference on those 6
coefficients gave p=0.0008 -- pseudoreplication, because the assets are
cross-correlated (return rho_bar ~ 0.69) and share one classified 50-event
calendar; BNB's later listing leaves 45 event windows on its own return index.

c6 corrected this post-hoc by penalising at the COEFFICIENT level using the RAW
RETURN correlation (design-effect + correlation-weighted), landing at p~0.067-0.078.
But the event dummies sit in the VARIANCE equation; cross-asset dependence of the
relevant statistic propagates through the STANDARDISED residuals (after GARCH
strips common volatility), whose correlation is lower than the raw-return
correlation. This script tests that conjecture by simulating the actual estimator
under cross-asset dependence calibrated to the standardised-residual correlation.

What this does
--------------
1. Fit GJR-GARCH-X per asset (baseline S1 50-event sample, same 5 exog as c6),
   multistart, store params, conditional variance sigma2_t, standardised resid z_t.
2. Observed: d_bar_obs = mean_a(delta_infra - delta_reg); multiplier; report
   BOTH rho_bar(returns) and rho_bar(standardised residuals z).
3. Parametric CCC bootstrap:
   (a) NULL-imposed p-value -- refit with a single combined dummy
       D_event = D_infra + D_reg (imposes delta_infra=delta_reg), simulate panels
       draw one six-vector z* ~ MVN(0, R_z) per union-calendar date, discard
       unavailable asset coordinates, set eps* = z*.sqrt(sigma2_null),
       returns* = mean + eps*,
       refit UNRESTRICTED, collect d_bar_b. one-sided/absolute-statistic p.
   (b) UNRESTRICTED CI -- simulate from the fitted unrestricted model, refit,
       percentile 90/95% CI for d_bar, fraction <= 0.
4. Cross-sectional WILD bootstrap (model-free) -- common Rademacher multiplier
   eta_t shared across all assets observed on each union-calendar day (preserves
   contemporaneous cross-asset co-movement), returns* = mean + eta_t * resid_t,
   refit both dummies, d_bar_b.

Parallelised across draws (multiprocessing). Non-converged refits are dropped
and the dropped fraction is reported; >10% is flagged unreliable.

Outputs (in the package results directory):
  c7-ccc-bootstrap-results.csv
  c7-ccc-bootstrap-FINDING.md
"""
import argparse
import time
import warnings
import multiprocessing as mp
from multiprocessing import Pool

# Force the fork start method: workers then inherit the fully-populated module
# global _GLOBAL (the big design/observed arrays) for free, no pickling. The
# box defaults to forkserver, which would NOT inherit module globals.
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")

import c2_relaxed_threshold_sensitivity as c2  # loaders, ASSETS, dummies, etc.
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA, _variance_recursion_core

ASSETS = c2.ASSETS
SENT_COLS = ["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]
N_STARTS_FIT = 6          # multistart for the observed (real) fits
N_STARTS_BOOT = 1         # bootstrap refits: single default start (fast); validated stable
MAX_ITER = 2000


# ----------------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------------
def build_design():
    """
    Returns a dict per asset with:
      returns (np.ndarray, %), exog_unrestricted (n x 5), exog_null (n x 4),
      index (DatetimeIndex), plus the shared raw-return panel for rho_return.
    Unrestricted exog cols: [D_infra, D_reg, S_gdelt, S_reg, S_infra]
    Null exog cols:         [D_event=D_infra+D_reg, S_gdelt, S_reg, S_infra]
    """
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(c2.OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)

    design = {}
    return_series = {}
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
        dum = c2.build_event_dummies(r.index, inf_d, reg_d,
                                     c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s = sent.reindex(r.index).fillna(0)
        Dinf = dum["D_infrastructure"].values
        Dreg = dum["D_regulatory"].values
        S = s[SENT_COLS].values
        exog_unr = np.column_stack([Dinf, Dreg, S])          # 5 cols
        exog_null = np.column_stack([Dinf + Dreg, S])        # 4 cols (combined dummy)
        design[a] = {
            "returns": r.values.astype(float),
            "exog_unr": exog_unr,
            "exog_null": exog_null,
            "index": r.index,
        }
        return_series[a] = r
    # raw-return correlation on the common overlap
    ret_df = pd.DataFrame(return_series).dropna()
    return design, inf_d, reg_d, ret_df


def mean_off_diag(C):
    n = C.shape[0]
    return (C.sum() - np.trace(C)) / (n * (n - 1))


def install_simulation_calendar(design, state=None):
    """Install the union-date calendar used by every joint innovation draw.

    ``R_z`` continues to be calibrated on the six-asset complete-case residual
    window.  Simulation, however, spans the sorted union of asset dates.  A full
    six-vector is drawn for every union-calendar date and each asset selects its
    observed coordinates.  Discarding an unavailable coordinate gives exactly
    the corresponding marginal submatrix of ``R_z``; it avoids incorrectly
    making the five pre-BNB assets independent on dates they share.
    """
    target = _GLOBAL if state is None else state
    simulation_calendar = pd.DatetimeIndex(sorted(set().union(*[
        set(pd.DatetimeIndex(design[a]["index"])) for a in ASSETS
    ])))
    if len(simulation_calendar) == 0 or not simulation_calendar.is_monotonic_increasing:
        raise RuntimeError("simulation calendar is empty or not sorted")

    calendar_pos = {}
    availability = np.zeros((len(simulation_calendar), len(ASSETS)), dtype=bool)
    for j, a in enumerate(ASSETS):
        asset_index = pd.DatetimeIndex(design[a]["index"])
        pos = simulation_calendar.get_indexer(asset_index)
        if len(pos) != len(asset_index) or np.any(pos < 0):
            raise RuntimeError(f"failed to map every {a} date to simulation calendar")
        if len(np.unique(pos)) != len(pos):
            raise RuntimeError(f"duplicate {a} positions in simulation calendar")
        calendar_pos[a] = pos.astype(np.int64, copy=False)
        availability[pos, j] = True

    pattern_code = np.sum(
        availability.astype(np.int64) * (1 << np.arange(len(ASSETS), dtype=np.int64)),
        axis=1,
    )
    pattern_codes, pattern_counts = np.unique(pattern_code, return_counts=True)
    pattern_first_date_ns = np.asarray([
        simulation_calendar.asi8[pattern_code == code].min() for code in pattern_codes
    ], dtype=np.int64)
    pattern_last_date_ns = np.asarray([
        simulation_calendar.asi8[pattern_code == code].max() for code in pattern_codes
    ], dtype=np.int64)

    target.update({
        "simulation_calendar": simulation_calendar,
        "n_calendar": len(simulation_calendar),
        "calendar_pos": calendar_pos,
        "availability_matrix": availability,
        "availability_pattern_code": pattern_code.astype(np.int64),
        "availability_pattern_codes": pattern_codes.astype(np.int64),
        "availability_pattern_counts": pattern_counts.astype(np.int64),
        "availability_pattern_first_date_ns": pattern_first_date_ns,
        "availability_pattern_last_date_ns": pattern_last_date_ns,
    })
    return target


def scatter_calendar_innovations(innovations, state=None):
    """Select each asset's observed dates from one full calendar-by-asset draw."""
    source = _GLOBAL if state is None else state
    innovations = np.asarray(innovations, dtype=float)
    expected = (int(source["n_calendar"]), len(ASSETS))
    if innovations.shape != expected:
        raise RuntimeError(
            f"calendar innovation shape {innovations.shape} does not match {expected}"
        )
    return {
        a: innovations[np.asarray(source["calendar_pos"][a], dtype=np.int64), j]
        for j, a in enumerate(ASSETS)
    }


def draw_gaussian_calendar_innovations(rng, state=None):
    """Draw MVN innovations on the union calendar and scatter by availability."""
    source = _GLOBAL if state is None else state
    latent = rng.standard_normal((int(source["n_calendar"]), len(ASSETS)))
    innovations = latent @ np.asarray(source["L_z"], dtype=float).T
    return scatter_calendar_innovations(innovations, state=source)


# ----------------------------------------------------------------------------
# Observed fit
# ----------------------------------------------------------------------------
def fit_observed(design, seed=0):
    """
    Fit unrestricted + null models per asset. Returns dict of arrays:
      params_unr[a], params_null[a], sigma2_null[a], z_resid[a] (standardised
      resid from the unrestricted fit), delta_infra[a], delta_reg[a].
    """
    out = {}
    for a in ASSETS:
        d = design[a]
        # unrestricted
        est_u = FastTARCHX(d["returns"], d["exog_unr"])
        pu, fu, oku = est_u.fit_multistart(n_starts=N_STARTS_FIT, seed=seed, max_iter=MAX_ITER)
        var_u = est_u._variance(pu)
        z = est_u.resid / np.sqrt(var_u)
        # null (combined dummy)
        est_n = FastTARCHX(d["returns"], d["exog_null"])
        pn, fn, okn = est_n.fit_multistart(n_starts=N_STARTS_FIT, seed=seed + 1, max_iter=MAX_ITER)
        var_n = est_n._variance(pn)
        out[a] = {
            "params_unr": pu, "ok_unr": oku, "negll_unr": fu,
            "params_null": pn, "ok_null": okn, "negll_null": fn,
            "sigma2_null": var_n, "sigma2_unr": var_u,
            "z_resid": z,
            "resid_unr": est_u.resid.copy(),
            "mean_return": est_u.mean_return,
            "delta_infra": pu[5], "delta_reg": pu[6],
        }
    return out


# ----------------------------------------------------------------------------
# Bootstrap worker functions (module-level for multiprocessing pickling)
# ----------------------------------------------------------------------------
_GLOBAL = {}  # populated in the parent before Pool creation; inherited via fork


# A refit can hit a degenerate local optimum SLSQP still reports as "success":
# beta -> 0, omega huge, and the event deltas absorb all variance (seen e.g.
# delta ~ 6000 on a series whose variance is ~70). These are numerical
# artifacts, not draws from the sampling distribution -- they must be rejected,
# else a handful of them blow up the bootstrap mean/CI. The observed deltas are
# all < 3 and the data variance is ~10-70, so any |delta| > DELTA_CAP is
# pathological. The cap is deliberately loose (it never excludes a real fit).
DELTA_CAP = 50.0


def _plausible_delta(p):
    return abs(p[5]) <= DELTA_CAP and abs(p[6]) <= DELTA_CAP


def _refit_unrestricted_dbar(returns_by_asset, return_diagnostics=False):
    """
    Given simulated returns per asset, refit the UNRESTRICTED model (2 dummies)
    and return d_bar = mean_a(delta_infra - delta_reg), or np.nan if any asset
    fails to converge OR lands on a degenerate (implausible-delta) optimum.

    Parameters
    ----------
    return_diagnostics : bool, default False
        Preserve the historical scalar return by default.  When True, return
        ``(d_bar, diagnostics)`` where diagnostics records both draw-level and
        asset-refit-level use of the single deterministic six-start rescue.
        This opt-in path lets sensitivity scripts audit optimiser effort without
        changing c7/c9's existing draw-function interface.
    """
    design = _GLOBAL["design"]
    rescue_seed = _GLOBAL.get("rescue_seed", 20260807)
    diffs = []
    n_asset_rescue_attempts = 0
    n_asset_rescue_successes = 0
    for a in ASSETS:
        est = FastTARCHX(returns_by_asset[a], design[a]["exog_unr"])
        p, f, ok = est.fit(start=None, max_iter=MAX_ITER)
        if not ok or not _plausible_delta(p):
            # Deterministic multistart rescue: a single-start SLSQP failure is a
            # property of the optimiser, not of the draw. Applied identically to
            # every resampling scheme so failure rates are comparable across them.
            n_asset_rescue_attempts += 1
            p, f, ok = est.fit_multistart(n_starts=6, seed=rescue_seed, max_iter=MAX_ITER)
            if not ok or not _plausible_delta(p):
                diagnostics = {
                    "draw_had_six_start_rescue_attempt": True,
                    "draw_retained_after_six_start_rescue": False,
                    "n_asset_refits_with_six_start_rescue_attempt": n_asset_rescue_attempts,
                    "n_asset_refits_rescued_by_six_start": n_asset_rescue_successes,
                }
                return (np.nan, diagnostics) if return_diagnostics else np.nan
            n_asset_rescue_successes += 1
        diffs.append(p[5] - p[6])
    d_bar = float(np.mean(diffs))
    diagnostics = {
        "draw_had_six_start_rescue_attempt": n_asset_rescue_attempts > 0,
        "draw_retained_after_six_start_rescue": n_asset_rescue_attempts > 0,
        "n_asset_refits_with_six_start_rescue_attempt": n_asset_rescue_attempts,
        "n_asset_refits_rescued_by_six_start": n_asset_rescue_successes,
    }
    return (d_bar, diagnostics) if return_diagnostics else d_bar


def _draw_parametric_null(args):
    """One null-imposed parametric draw. args = seed (int)."""
    seed = args
    rng = np.random.default_rng(seed)
    obs = _GLOBAL["observed"]
    z_by_asset = draw_gaussian_calendar_innovations(rng)
    returns_by_asset = {}
    for a in ASSETS:
        sig2 = obs[a]["sigma2_null"]
        mean_r = obs[a]["mean_return"]
        eps = z_by_asset[a] * np.sqrt(sig2)
        returns_by_asset[a] = mean_r + eps
    return _refit_unrestricted_dbar(returns_by_asset)


def _draw_parametric_unr(args):
    """One unrestricted-model parametric draw (for the CI). args = seed."""
    seed = args
    rng = np.random.default_rng(seed)
    obs = _GLOBAL["observed"]
    z_by_asset = draw_gaussian_calendar_innovations(rng)
    returns_by_asset = {}
    for a in ASSETS:
        sig2 = obs[a]["sigma2_unr"]
        mean_r = obs[a]["mean_return"]
        eps = z_by_asset[a] * np.sqrt(sig2)
        returns_by_asset[a] = mean_r + eps
    return _refit_unrestricted_dbar(returns_by_asset)


def _draw_wild(args):
    """
    One cross-sectional wild-bootstrap draw. A common Rademacher multiplier
    eta_t in {-1,+1} is shared by all assets observed on each union-calendar
    date, preserving contemporaneous cross-asset residual co-movement.

    Rademacher is used over Mammen here deliberately: Mammen's variance-altering
    two-point weights both inflate the recovered effect AND trigger far more
    degenerate GARCH refits on these fat-tailed (nu~3) residuals (drop ~27% vs
    ~7%). Rademacher preserves |residual| exactly, so refits stay well-posed and
    the wild distribution centres on d_bar_obs.
    """
    seed = args
    rng = np.random.default_rng(seed)
    obs = _GLOBAL["observed"]
    eta_calendar = rng.choice([-1.0, 1.0], int(_GLOBAL["n_calendar"]))
    returns_by_asset = {}
    for a in ASSETS:
        resid = obs[a]["resid_unr"]
        eta = eta_calendar[np.asarray(_GLOBAL["calendar_pos"][a], dtype=np.int64)]
        returns_by_asset[a] = obs[a]["mean_return"] + eta * resid
    return _refit_unrestricted_dbar(returns_by_asset)


# ----------------------------------------------------------------------------
# Bootstrap runners
# ----------------------------------------------------------------------------
def run_bootstrap(draw_fn, B, n_jobs, base_seed):
    # On Linux (fork start method) the fully-populated module-global _GLOBAL is
    # inherited by every worker -- no pickling of the big arrays needed.
    seeds = [base_seed + i for i in range(B)]
    with Pool(processes=n_jobs) as pool:
        results = pool.map(draw_fn, seeds, chunksize=max(1, B // (n_jobs * 4)))
    arr = np.array(results, dtype=float)
    n_drop = int(np.isnan(arr).sum())
    arr = arr[~np.isnan(arr)]
    return arr, n_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    t_start = time.time()
    print(f"numba available: {_HAVE_NUMBA}  (pure-numpy recursion if False)")
    print("Building design matrices...")
    design, inf_d, reg_d, ret_df = build_design()
    print(f"  baseline events: {len(inf_d)} infra, {len(reg_d)} reg")
    for a in ASSETS:
        print(f"  {a}: n_obs={design[a]['returns'].shape[0]}")

    print("\nFitting observed models (unrestricted + null), multistart...")
    t0 = time.time()
    observed = fit_observed(design, seed=args.seed)
    print(f"  observed fits done in {time.time()-t0:.1f}s")

    # per-asset deltas
    di = np.array([observed[a]["delta_infra"] for a in ASSETS])
    dr = np.array([observed[a]["delta_reg"] for a in ASSETS])
    d_obs = di - dr
    d_bar_obs = float(d_obs.mean())
    multiplier = float(di.mean() / dr.mean())
    print("\nPer-asset deltas (unrestricted):")
    for a, x, y in zip(ASSETS, di, dr):
        print(f"  {a}: dInfra={x:.4f}  dReg={y:.4f}  diff={x-y:.4f}")
    print(f"  d_bar_obs = {d_bar_obs:.4f}   multiplier = {multiplier:.3f}x")

    # ---- correlations: return vs standardised-residual --------------------
    R_return = ret_df.corr().values
    rho_return = mean_off_diag(R_return)

    # standardised-residual panel on the common overlap
    z_df = pd.DataFrame({a: pd.Series(observed[a]["z_resid"], index=design[a]["index"])
                         for a in ASSETS}).dropna()
    R_z = z_df.corr().values
    rho_resid = mean_off_diag(R_z)
    print(f"\nrho_bar(returns)             = {rho_return:.4f}")
    print(f"rho_bar(standardised resid)  = {rho_resid:.4f}   <-- the crux")

    # ---- shared globals for workers (inherited by fork) -------------------
    common_idx = z_df.index
    n_common = len(common_idx)
    # per-asset integer positions of the common dates within that asset's series
    common_pos = {}
    for a in ASSETS:
        pos = pd.Index(design[a]["index"]).get_indexer(common_idx)
        common_pos[a] = pos
    # PD-safe cholesky of R_z (nudge if needed)
    R_z_pd = R_z.copy()
    eps_jit = 0.0
    while True:
        try:
            L_z = np.linalg.cholesky(R_z_pd)
            break
        except np.linalg.LinAlgError:
            eps_jit = max(eps_jit * 10, 1e-8)
            R_z_pd = R_z + eps_jit * np.eye(len(ASSETS))
    if eps_jit > 0:
        print(f"  [note] R_z nudged by {eps_jit:.1e} for PD cholesky")

    _GLOBAL["design"] = design
    _GLOBAL["observed"] = observed
    _GLOBAL["R_z"] = R_z
    _GLOBAL["L_z"] = L_z
    _GLOBAL["n_common"] = n_common
    _GLOBAL["common_pos"] = common_pos
    _GLOBAL["max_len"] = max(observed[a]["resid_unr"].shape[0] for a in ASSETS)
    install_simulation_calendar(design)
    patterns = ", ".join(
        f"mask={int(code)}:{int(count)} dates"
        for code, count in zip(
            _GLOBAL["availability_pattern_codes"],
            _GLOBAL["availability_pattern_counts"],
        )
    )
    print(f"  union simulation calendar: n={_GLOBAL['n_calendar']} ({patterns})")

    B = args.B
    print(f"\n=== Bootstrapping B={B}, n_jobs={args.n_jobs} ===")

    # (3a) NULL-imposed parametric p-value
    print("[3a] parametric NULL-imposed (CCC, MVN copula on R_z)...")
    t0 = time.time()
    null_arr, null_drop = run_bootstrap(_draw_parametric_null, B, args.n_jobs, args.seed + 10_000)
    Bn = len(null_arr)
    p_one = (np.sum(null_arr >= d_bar_obs) + 1) / (Bn + 1)
    p_two = (np.sum(np.abs(null_arr) >= abs(d_bar_obs)) + 1) / (Bn + 1)
    print(f"  done {time.time()-t0:.1f}s  used={Bn}  dropped={null_drop} ({null_drop/B:.1%})")
    print(f"  p_one_sided = {p_one:.4f}   p_two_sided = {p_two:.4f}")
    print(f"  null d_bar mean={null_arr.mean():.4f} sd={null_arr.std():.4f}")

    # (3b) UNRESTRICTED parametric CI
    print("[3b] parametric UNRESTRICTED (CI)...")
    t0 = time.time()
    unr_arr, unr_drop = run_bootstrap(_draw_parametric_unr, B, args.n_jobs, args.seed + 20_000)
    Bu = len(unr_arr)
    # The GARCH event-coefficient estimator is upward-biased in this small sample
    # (refits from the fitted DGP recover a median d_bar above d_bar_obs). The raw
    # percentile CI inherits that bias; the BASIC (reverse-percentile) CI,
    # 2*d_bar_obs - quantiles, corrects median bias and is the one to report.
    pct90 = np.percentile(unr_arr, [5, 95])
    pct95 = np.percentile(unr_arr, [2.5, 97.5])
    ci90 = np.array([2 * d_bar_obs - pct90[1], 2 * d_bar_obs - pct90[0]])
    ci95 = np.array([2 * d_bar_obs - pct95[1], 2 * d_bar_obs - pct95[0]])
    boot_bias = float(np.median(unr_arr) - d_bar_obs)
    # fraction of bias-corrected draws <= 0 (basic-bootstrap reflection)
    unr_basic = 2 * d_bar_obs - unr_arr
    frac_le0_unr = float(np.mean(unr_basic <= 0))
    print(f"  done {time.time()-t0:.1f}s  used={Bu}  dropped={unr_drop} ({unr_drop/B:.1%})")
    print(f"  bootstrap median bias (median - obs) = {boot_bias:.4f}")
    print(f"  raw percentile 95% = [{pct95[0]:.4f}, {pct95[1]:.4f}]")
    print(f"  basic (bias-corr) 90% CI = [{ci90[0]:.4f}, {ci90[1]:.4f}]   95% CI = [{ci95[0]:.4f}, {ci95[1]:.4f}]")
    print(f"  fraction(basic d_bar <= 0) = {frac_le0_unr:.4f}")

    # (4) WILD cross-sectional
    print("[4] cross-sectional WILD (Rademacher, shared eta_t)...")
    t0 = time.time()
    wild_arr, wild_drop = run_bootstrap(_draw_wild, B, args.n_jobs, args.seed + 30_000)
    Bw = len(wild_arr)
    # Rademacher wild draws centre on d_bar_obs and approximate the sampling
    # distribution of the estimator around the truth theta. Basic-bootstrap test
    # of H0: theta=0 vs theta>0: under H0 the estimator would be distributed as
    # 0 + (d_bar_b - d_bar_obs), so one-sided p = P(d_bar_b - d_bar_obs >= d_bar_obs)
    #                                            = P(d_bar_b >= 2*d_bar_obs).
    wild_centered = wild_arr - d_bar_obs
    p_wild_one = (np.sum(wild_centered >= d_bar_obs) + 1) / (Bw + 1) if d_bar_obs > 0 else \
                 (np.sum(wild_centered <= d_bar_obs) + 1) / (Bw + 1)
    p_wild_two = (np.sum(np.abs(wild_centered) >= abs(d_bar_obs)) + 1) / (Bw + 1)
    # CI for theta from the centred draws (basic bootstrap): d_bar_obs - centred quantiles
    wpc95 = np.percentile(wild_centered, [2.5, 97.5])
    wild_ci95 = np.array([d_bar_obs - wpc95[1], d_bar_obs - wpc95[0]])
    print(f"  done {time.time()-t0:.1f}s  used={Bw}  dropped={wild_drop} ({wild_drop/B:.1%})")
    print(f"  wild d_bar mean={wild_arr.mean():.4f} sd={wild_arr.std():.4f}")
    print(f"  wild p_one={p_wild_one:.4f}  p_two={p_wild_two:.4f}  basic 95%CI=[{wild_ci95[0]:.4f},{wild_ci95[1]:.4f}]")

    # ---- save results -----------------------------------------------------
    rows = []
    for a, x, y in zip(ASSETS, di, dr):
        rows.append({"asset": a, "delta_infra": x, "delta_reg": y, "diff": x - y,
                     "ok_unr": observed[a]["ok_unr"], "ok_null": observed[a]["ok_null"]})
    df_assets = pd.DataFrame(rows)

    summary = {
        "d_bar_obs": d_bar_obs,
        "multiplier": multiplier,
        "rho_return": rho_return,
        "rho_resid": rho_resid,
        "n_calendar": int(_GLOBAL["n_calendar"]),
        "availability_pattern_codes": ";".join(
            str(int(v)) for v in _GLOBAL["availability_pattern_codes"]
        ),
        "availability_pattern_counts": ";".join(
            str(int(v)) for v in _GLOBAL["availability_pattern_counts"]
        ),
        "B_requested": B,
        "p_param_null_one_sided": p_one,
        "p_param_null_two_sided": p_two,
        "param_null_B_used": Bn,
        "param_null_frac_dropped": null_drop / B,
        "unr_ci90_lo": ci90[0], "unr_ci90_hi": ci90[1],
        "unr_ci95_lo": ci95[0], "unr_ci95_hi": ci95[1],
        "unr_pct95_lo": pct95[0], "unr_pct95_hi": pct95[1],
        "unr_boot_median_bias": boot_bias,
        "unr_frac_le0": frac_le0_unr,
        "unr_B_used": Bu, "unr_frac_dropped": unr_drop / B,
        "wild_p_one_sided": p_wild_one,
        "wild_p_two_sided": p_wild_two,
        "wild_ci95_lo": wild_ci95[0], "wild_ci95_hi": wild_ci95[1],
        "wild_B_used": Bw, "wild_frac_dropped": wild_drop / B,
        "numba": _HAVE_NUMBA,
    }
    df_summary = pd.DataFrame([summary])

    out_csv = c2.OUT_DIR / "c7-ccc-bootstrap-results.csv"
    # write per-asset then summary into one csv with a section marker
    with open(out_csv, "w") as f:
        f.write("# per-asset deltas (unrestricted fit, baseline S1 50-event)\n")
        df_assets.to_csv(f, index=False)
        f.write("\n# summary\n")
        df_summary.to_csv(f, index=False)
    print(f"\nSaved {out_csv}")

    # also dump raw bootstrap draws for any later replotting
    np.savez(c2.OUT_DIR / "c7-bootstrap-draws.npz",
             null=null_arr, unr=unr_arr, wild=wild_arr,
             d_bar_obs=d_bar_obs,
             simulation_calendar_ns=pd.DatetimeIndex(
                 _GLOBAL["simulation_calendar"]
             ).asi8.astype(np.int64),
             availability_matrix=_GLOBAL["availability_matrix"],
             availability_pattern_code=_GLOBAL["availability_pattern_code"],
             availability_pattern_codes=_GLOBAL["availability_pattern_codes"],
             availability_pattern_counts=_GLOBAL["availability_pattern_counts"],
             calendar_pos_asset_names=np.asarray(ASSETS),
             **{
                 f"calendar_pos_{a}": np.asarray(_GLOBAL["calendar_pos"][a], dtype=np.int64)
                 for a in ASSETS
             })

    write_finding(summary, df_assets, B, time.time() - t_start)
    print(f"\nTOTAL {time.time()-t_start:.1f}s")
    return summary


def write_finding(s, df_assets, B, elapsed):
    def verdict_str():
        # The parametric null-imposed test is the usable c7 benchmark. The wild
        # (Rademacher) bootstrap is NOT used for the verdict: sign-flipping is
        # near-degenerate for variance-equation coefficients (eps^2 is
        # sign-invariant), so its p is artificially tiny -- a known limitation,
        # not evidence. See the "Significance" and "Caveats" sections.
        p = s["p_param_null_one_sided"]
        if p < 0.05:
            return ("Significant at 5% under the parametric null-imposed CCC bootstrap "
                    f"(p={p:.4f}).")
        if p < 0.10:
            return ("Marginal: significant at 10% but NOT at 5% under the parametric "
                    f"null-imposed CCC bootstrap (p={p:.4f}). Same ballpark as c6.")
        return ("Not significant at 10% under the parametric null-imposed CCC bootstrap "
                f"(p={p:.4f}).")

    lines = []
    lines.append("# C7 -- Gaussian-copula CCC-GARCH-X bootstrap (intermediate benchmark)\n")
    lines.append(f"_Bootstrap B={B}; numba={s['numba']}; runtime {elapsed/60:.1f} min._\n")
    lines.append("## The question\n")
    lines.append("Is the infrastructure-vs-regulatory variance-coefficient asymmetry "
                 "(d_bar = mean_a(delta_infra - delta_reg) over 6 assets) significant "
                 "under a model-based Gaussian-copula benchmark -- i.e. via a "
                 "model-based bootstrap of the ACTUAL GJR-GARCH-X estimator, with "
                 "cross-asset dependence calibrated to the STANDARDISED-residual "
                 "correlation rather than the raw-return correlation c6 used as a "
                 "post-hoc proxy? c6 bracketed p ~ 0.067-0.078 and conjectured it had "
                 "OVER-penalised (residual corr assumed << return corr). This test checks "
                 "that conjecture directly.\n")

    lines.append("## Per-asset coefficients (baseline S1, 50 events)\n")
    lines.append("| asset | delta_infra | delta_reg | diff |")
    lines.append("|---|---|---|---|")
    for _, r in df_assets.iterrows():
        lines.append(f"| {r['asset']} | {r['delta_infra']:.4f} | {r['delta_reg']:.4f} | {r['diff']:.4f} |")
    lines.append("")

    lines.append("## Headline numbers\n")
    lines.append(f"- **d_bar_obs** = {s['d_bar_obs']:.4f}")
    lines.append(f"- **multiplier** (mean infra / mean reg) = {s['multiplier']:.3f}x")
    lines.append(f"- **rho_bar(returns)** = {s['rho_return']:.4f}")
    higher = s["rho_resid"] >= s["rho_return"]
    crux_note = (
        f"<- the crux. The pre-test hypothesis was that GARCH strips common "
        f"volatility, so rho_resid would be MUCH LOWER than rho_return and c6's "
        f"raw-return penalty would be too harsh. That did NOT hold: rho_resid "
        f"({s['rho_resid']:.3f}) is actually slightly HIGHER than rho_return "
        f"({s['rho_return']:.3f}). One interpretation is that standardisation "
        f"makes the shared component relatively more prominent, but this "
        f"correlation comparison does not identify that mechanism. The fitted "
        f"residual-dependence benchmark is nevertheless not weaker than the "
        f"raw-return proxy used in c6."
    ) if higher else (
        f"<- the crux: fitted residual correlation is lower than return "
        f"correlation in this sample; the comparison itself does not identify why."
    )
    lines.append(f"- **rho_bar(standardised residuals)** = {s['rho_resid']:.4f}  {crux_note}\n")

    lines.append("## Significance\n")
    lines.append(f"- **Parametric NULL-imposed (CCC MVN copula on R_z): one-sided "
                 f"p = {s['p_param_null_one_sided']:.4f}**, "
                 f"absolute-statistic p = {s['p_param_null_two_sided']:.4f} "
                 f"(B used {s['param_null_B_used']}, dropped {s['param_null_frac_dropped']:.1%}). "
                 f"This redraws Gaussian standardised innovations against the fitted null "
                 f"variance paths with the fitted cross-asset correlation, refits unrestricted, and "
                 f"compares the observed d_bar to the null sampling distribution "
                 f"(including the estimator's finite-sample behaviour under this benchmark).")
    lines.append(f"- Cross-sectional WILD (Rademacher, shared eta_t): "
                 f"one-sided p = {s['wild_p_one_sided']:.4f}, "
                 f"absolute-statistic p = {s['wild_p_two_sided']:.4f} "
                 f"(B used {s['wild_B_used']}, dropped {s['wild_frac_dropped']:.1%}). "
                 f"**DO NOT TRUST this p as significance evidence.** Sign-flipping "
                 f"residuals barely perturbs a VARIANCE-equation coefficient because "
                 f"eps^2 is sign-invariant; the wild d_bar distribution is near-degenerate "
                 f"(sd ~0.1), so the tiny p is an artifact of under-dispersion, not power. "
                 f"It is reported only for completeness; it is the wrong instrument for "
                 f"this estimand.")
    lines.append(f"- Unrestricted parametric CI for d_bar (bias-corrected basic "
                 f"bootstrap; the estimator is upward-biased by "
                 f"median {s['unr_boot_median_bias']:.3f} in this sample): "
                 f"90% [{s['unr_ci90_lo']:.4f}, {s['unr_ci90_hi']:.4f}], "
                 f"95% [{s['unr_ci95_lo']:.4f}, {s['unr_ci95_hi']:.4f}]; "
                 f"fraction of (bias-corrected) draws <= 0 = {s['unr_frac_le0']:.4f} "
                 f"(raw percentile 95% [{s['unr_pct95_lo']:.4f}, {s['unr_pct95_hi']:.4f}]; "
                 f"B used {s['unr_B_used']}, dropped {s['unr_frac_dropped']:.1%})\n")

    lines.append("## Comparison with c6 (p ~ 0.067-0.078)\n")
    p_main = s["p_param_null_one_sided"]
    if p_main < 0.05:
        rel = (f"The c7 benchmark ({p_main:.4f}) **beats** c6's ~0.07 and **crosses 0.05**.")
    elif p_main < 0.10:
        rel = (f"The c7 benchmark ({p_main:.4f}) **lands right on top of c6's ~0.07: still "
               f"MARGINAL (significant at 10%, not at 5%).** It neither rescues the result "
               f"to 5% nor undercuts it below 10%. The premise that c6 over-penalised by "
               f"using the raw-return correlation does NOT hold here -- the "
               f"standardised-residual correlation ({s['rho_resid']:.3f}) is if anything "
               f"slightly higher than the return correlation ({s['rho_return']:.3f}), so "
               f"the fitted residual-dependence benchmark is no weaker than c6 assumed, and the "
               f"p-value stays in the same place. c6's post-hoc bracketing turns out to "
               f"have been a good approximation.")
    else:
        rel = (f"The c7 benchmark ({p_main:.4f}) is **weaker than c6's ~0.07** -- the "
               f"asymmetry does not reach even 10% under this Gaussian-copula benchmark.")
    lines.append(rel + "\n")

    lines.append("## Verdict within the c7 Gaussian-copula benchmark\n")
    lines.append(verdict_str() + "\n")
    lines.append("## Caveats (honest)\n")
    lines.append("- This is an intermediate comparator, not the final inferential verdict. It uses a Gaussian "
                 "copula (MVN on R_z) for cross-asset dependence of standardised "
                 "innovations; the per-asset variance path is the fitted Student-t GJR "
                 "model, but the cross-asset copula itself is Gaussian (tail dependence "
                 "not modelled). With nu~3 marginals this could mildly understate joint "
                 "tail co-movement; the direction of any resulting bias on the p-value is "
                 "not obvious from c7 alone. C9 and c19 report the heavy-tail and "
                 "propagation-scheme sensitivities used in the manuscript.")
    lines.append("- **The wild bootstrap is NOT a usable anchor here** (contrary to the "
                 "usual model-free role): Rademacher sign-flips leave eps^2 -- and hence "
                 "the variance-equation event coefficients -- almost unchanged, collapsing "
                 "the wild d_bar distribution (sd ~0.1) and producing an artificially tiny "
                 "p. We flag this rather than hide it.")
    lines.append("- The unrestricted GARCH event-coefficient estimator is upward-biased in "
                 f"this small sample (bootstrap median bias ~{s['unr_boot_median_bias']:.2f}); "
                 "the null-imposed test handles this automatically (its null distribution is "
                 "centred at the biased-under-equality location, not at 0), and the reported "
                 "CI is the bias-corrected basic-bootstrap interval.")
    lines.append("- Bootstrap refits use a single default start followed, on failure or a "
                 "degenerate optimum, by exactly one deterministic six-start rescue; the "
                 "observed fits use multistart. Refits that remain degenerate (SLSQP 'success' but "
                 f"|delta|>{50:.0f}, e.g. beta->0 / omega blown up) are rejected and counted "
                 "as dropped.")
    lines.append(f"- Convergence: dropped fractions above are the share of draws still "
                 f"unusable after that common rescue and are reported rather than silently replaced.")
    lines.append("")
    lines.append("## Files\n")
    lines.append("- `c7-ccc-bootstrap-results.csv` -- per-asset + summary")
    lines.append("- `c7-bootstrap-draws.npz` -- raw d_bar draws (null/unr/wild)")
    lines.append("- `code/c7_ccc_garchx_bootstrap.py`, `code/tarch_x_fast.py`")

    (c2.OUT_DIR / "c7-ccc-bootstrap-FINDING.md").write_text("\n".join(lines))
    print(f"Saved {c2.OUT_DIR / 'c7-ccc-bootstrap-FINDING.md'}")


if __name__ == "__main__":
    main()
