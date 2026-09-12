"""
C10: Monte-Carlo size-distortion study for the inference ladder.
================================================================================

The paper's central methodological claim is that the headline t=4.768 / p=0.0008
(iid t-test on 6 per-asset event coefficients) is *pseudoreplication*: the six
assets are cross-correlated (rho ~ 0.69-0.70) and share one classified 50-event
calendar (BNB's later listing leaves 45 supported event windows), so a test
treating them as N=6 independent units OVER-REJECTS. The fixes ladder up to
design-effect correction, a Gaussian-copula CCC-GARCH-X bootstrap (c7), and a
Student-t-copula CCC-GARCH-X bootstrap (c9, the fixed-path heavy-tail benchmark).

This script turns that cautionary tale from an anecdote into a *demonstrated*
result. It simulates event-study panels under a TRUE NULL of NO differential
event effect (delta_infra = delta_reg) from a fixed-design DGP fitted to the data,
and measures the empirical REJECTION RATE (size) of each method at the 0.05 and
0.10 nominal levels. A correctly sized test rejects a true null with probability
alpha, up to Monte-Carlo error.

DGP (true null, fitted to the data)
-----------------------------------
* Per-asset GJR-GARCH-X variance dynamics, estimated under the NULL-imposed
  combined-dummy spec D_event = D_infra + D_reg (so delta_infra = delta_reg by
  construction: there is NO differential event effect in the truth).
* Cross-asset dependence at the STANDARDISED-residual correlation R_z
  (rho_resid ~ 0.70), via a Cholesky factor.
* Student-t(nu) innovations at each asset's FITTED nu (nu ~ 3.1-4.6), unit
  variance (rescaled by sqrt((nu-2)/nu); sigma2_t carries the scale).
* Joint tail dependence via a true multivariate-t copula (shared chi-square
  mixing at the median fitted nu) -- identical machinery to c9.
* The actual 26 infra / 24 reg event-label structure and the real event-window
  dummies are reused unchanged for every simulated panel; only the returns are
  re-simulated. So delta_infra and delta_reg are estimated against the SAME
  design the paper uses.

The inference ladder (each applied to every simulated panel)
------------------------------------------------------------
For each panel we refit the UNRESTRICTED model (two separate dummies) per asset,
recovering 6 (delta_infra, delta_reg) pairs, their per-asset model SEs (numerical
Hessian on the c9/c7 fast estimator's own log-likelihood), and
d_bar = mean_a(delta_infra - delta_reg).

  (i)   NAIVE iid t-test          : Welch t-test on the 6 delta_infra vs 6
                                     delta_reg as if iid (the paper's headline
                                     rule). Reject if p < alpha.
  (ii)  DESIGN-EFFECT corrected   : inflate the SE of mean_d by the design effect
                                     sqrt(1 + (N-1) rho_bar) using the cross-asset
                                     RETURN correlation (exactly c6's rule). t on
                                     N-1 df.
  (iii) GAUSSIAN-COPULA bootstrap  : the c7 rule. Reject if the panel's d_bar lies
                                     beyond the critical value of the Gaussian-
                                     copula null sampling distribution.
  (iv)  STUDENT-t-COPULA bootstrap : the c9 rule, with the heavy-tailed t-copula
                                     null distribution.

Internal calibration of (iii)/(iv) without a nested plug-in bootstrap
---------------------------------------------------------------------
A copula-bootstrap test rejects when the observed d_bar falls outside its null
sampling distribution. Here one reference distribution is calibrated from the
same fitted fixed-path DGP used for every outer panel, then applied to N
independent panels from that generator. This measures the decision rule's
self-consistency under a shared known calibration DGP. It does not reproduce the
stronger plug-in workflow in which the null DGP and critical values are
re-estimated separately for every panel; that claim would require a fully nested
O(N x B) design.

Self-validation
----------------
The Student-t-copula critical values and panels share a fitted DGP, so
near-nominal size is expected if the implementation is internally consistent.
That is reassuring but is not independent validation under the unknown true DGP.
We report the full size table, over-rejection factors, convergence screening,
and Monte-Carlo error.

Run:
    python c10_size_study.py --N 1000 --B-ref 4000 --n-jobs 22 \
        --fit-seed 12345 --mc-seed 20260618

Outputs (results/):
    c10-size-study-results.csv
    c10-size-study-metadata.csv
    c10-size-study-FINDING.md
    c10-size-study-draws.npz
"""
import argparse
import hashlib
import inspect
import time
import warnings
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path

# fork so workers inherit the populated module globals for free (no pickling of
# the big design arrays) -- same trick c7/c9 rely on.
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")

import c2_relaxed_threshold_sensitivity as c2
import c7_ccc_garchx_bootstrap as c7
import c9_tcopula_bootstrap as c9
from c9_tcopula_bootstrap import _student_t_calendar_innovations
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA

ASSETS = c2.ASSETS
N_STARTS_FIT = 6          # multistart for the observed (real, DGP-defining) fits
MAX_ITER = 2000
DELTA_CAP = c7.DELTA_CAP  # 50.0; reject degenerate refits (same guard as c7/c9)
FIT_SEED_DEFAULT = 12345
MC_SEED_DEFAULT = 20260618
RESCUE_SEED = 20260807
REFERENCE_SEED_OFFSET = 700_000
T_REFERENCE_SEED_OFFSET = 500_000
C10_SEED_SCHEMA_VERSION = 1
C10_ANALYSIS_CONTRACT_VERSION = 1

# Module-global state inherited by workers via fork.
_G = {}

_C9_HASH_KEYS = (
    "audit_design_sha256",
    "audit_fixed_null_dgp_sha256",
    "audit_refitter_sha256",
    "audit_simulator_sha256",
    "audit_analysis_stack_sha256",
)
_C9_EXACT_ARRAY_KEYS = (
    "audit_asset_names",
    "audit_null_params_by_asset",
    "audit_null_mean_returns",
    "audit_null_sigma2_concat",
    "audit_null_sigma2_offsets",
    "audit_nu_null_by_asset",
    "audit_nu_c_null",
    "audit_R_z",
    "audit_L_z",
    "audit_simulation_calendar_ns",
    "audit_calendar_pos_concat",
    "audit_calendar_pos_offsets",
    "audit_availability_matrix",
    "audit_availability_pattern_code",
    "audit_availability_pattern_codes",
    "audit_availability_pattern_counts",
    "audit_availability_pattern_first_date_ns",
    "audit_availability_pattern_last_date_ns",
)


def _scalar_text(value):
    """Return a stable string from a scalar NumPy archive field."""
    array = np.asarray(value)
    if array.shape != ():
        raise RuntimeError(f"expected scalar text field, got shape {array.shape}")
    return str(array.item())


def _sha256_file(path):
    """Hash a source/artifact file without depending on checkout paths."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_c9_baseline(local_audit, fit_seed):
    """Fail unless the locally fitted c10 DGP is exactly c9 baseline schema v2.

    This gate runs after the inexpensive observed-data fits but before any
    reference or outer Monte-Carlo draws.  Hash equality is supplemented by
    exact parameter/path/mapping equality so an archive cannot pass merely by
    carrying copied fingerprint strings.
    """
    path = c2.OUT_DIR / "c9-tcopula-draws-baseline.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "c10 requires the final c9 baseline archive before computation: "
            f"{path}"
        )
    required = {
        "seed_schema_version", "base_seed", "audit_spec", *_C9_HASH_KEYS,
        *_C9_EXACT_ARRAY_KEYS,
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(
                f"{path.name} lacks c9 schema-v2 provenance fields: {sorted(missing)}"
            )
        fixed = {key: archive[key] for key in required}

    if int(fixed["seed_schema_version"]) != 2:
        raise RuntimeError("c10 requires c9 seed_schema_version=2")
    if int(fixed["base_seed"]) != int(fit_seed):
        raise RuntimeError(
            "c10 fit seed does not match c9 baseline base/fit seed: "
            f"c10={fit_seed}, c9={int(fixed['base_seed'])}"
        )
    if _scalar_text(fixed["audit_spec"]) != "baseline":
        raise RuntimeError("c9 archive is not the baseline specification")

    for key in _C9_HASH_KEYS:
        local = _scalar_text(local_audit[key])
        committed = _scalar_text(fixed[key])
        if local != committed:
            raise RuntimeError(f"c10 local {key} differs from c9 baseline")
    for key in _C9_EXACT_ARRAY_KEYS:
        if not np.array_equal(np.asarray(local_audit[key]), np.asarray(fixed[key])):
            raise RuntimeError(f"c10 local {key} differs exactly from c9 baseline")

    return {
        "c9_baseline_archive": str(path),
        "c9_seed_schema_version": 2,
        **{key: _scalar_text(local_audit[key]) for key in _C9_HASH_KEYS},
    }


def setup_fitted_dgp(fit_seed=FIT_SEED_DEFAULT, require_c9=True):
    """Fit and install the baseline fixed-path DGP used by c10 and c10b."""
    design, inf_d, reg_d, ret_df = c7.build_design()
    observed = c7.fit_observed(design, seed=int(fit_seed))

    di_o = np.array([observed[a]["delta_infra"] for a in ASSETS])
    dr_o = np.array([observed[a]["delta_reg"] for a in ASSETS])
    d_bar_obs = float((di_o - dr_o).mean())
    multiplier = float(di_o.mean() / dr_o.mean())

    R_return = ret_df.corr().values
    rho_return = c7.mean_off_diag(R_return)
    z_df = pd.DataFrame({
        a: pd.Series(observed[a]["z_resid"], index=design[a]["index"])
        for a in ASSETS
    }).dropna()
    R_z = z_df.corr().values
    rho_resid = c7.mean_off_diag(R_z)
    common_idx = z_df.index
    common_pos = {
        a: pd.Index(design[a]["index"]).get_indexer(common_idx) for a in ASSETS
    }

    R_z_pd = R_z.copy()
    eps_jit = 0.0
    while True:
        try:
            L_z = np.linalg.cholesky(R_z_pd)
            break
        except np.linalg.LinAlgError:
            eps_jit = max(eps_jit * 10, 1e-8)
            R_z_pd = R_z + eps_jit * np.eye(len(ASSETS))

    _G.clear()
    _G.update({
        "design": design,
        "observed": observed,
        "R_z": R_z,
        "L_z": L_z,
        "n_common": len(common_idx),
        "common_pos": common_pos,
        "rescue_seed": RESCUE_SEED,
        "max_len": max(observed[a]["resid_unr"].shape[0] for a in ASSETS),
    })
    c7.install_simulation_calendar(design, state=_G)
    c7._GLOBAL.update(_G)
    nu_unr, nu_null = c9._install_nu(observed)
    _G.update({
        "nu_unr": nu_unr,
        "nu_null": nu_null,
        "nu_c_unr": float(np.median(nu_unr)),
        "nu_c_null": float(np.median(nu_null)),
    })
    c7._GLOBAL.update(_G)

    local_audit = c9._fixed_dgp_audit_payload("baseline")
    provenance = (
        _require_exact_c9_baseline(local_audit, fit_seed) if require_c9 else {
            "c9_baseline_archive": "MOCK_OR_TEST_ONLY",
            "c9_seed_schema_version": 2,
            **{key: _scalar_text(local_audit[key]) for key in _C9_HASH_KEYS},
        }
    )
    return {
        "design": design,
        "inf_d": inf_d,
        "reg_d": reg_d,
        "ret_df": ret_df,
        "observed": observed,
        "d_bar_obs": d_bar_obs,
        "multiplier": multiplier,
        "rho_return": rho_return,
        "rho_resid": rho_resid,
        "nu_null": np.asarray(nu_null, dtype=float),
        "nu_c_null": float(np.median(nu_null)),
        "audit_payload": local_audit,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Per-asset model SEs for the two event coefficients via a numerical Hessian of
# the fast estimator's OWN negative log-likelihood at the fitted point. We need
# SEs only for the iid (i) and design-effect (ii) ladder rungs; (iii)/(iv) are
# bootstrap and need no SEs. Restricting the Hessian to the 2 event-coef rows is
# both faster and more stable (the nuisance GARCH params are well-identified and
# their cross-curvature with the deltas is small in this model).
# ---------------------------------------------------------------------------
def _delta_ses(est, params, idx=(5, 6), h=1e-4):
    """
    SEs of params[idx] from the 2x2 sub-block of the numerical Hessian of the
    negative log-likelihood (= observed information). Returns (se_infra, se_reg)
    or (nan, nan) if the sub-block is not positive-definite / invertible.
    """
    p = np.asarray(params, dtype=float)
    f = est._neg_loglik
    k = len(idx)
    H = np.zeros((k, k))
    f0 = f(p)
    if not np.isfinite(f0):
        return np.nan, np.nan
    # central second differences on the 2-coef sub-block
    for a in range(k):
        ia = idx[a]
        # diagonal
        pp = p.copy(); pp[ia] += h
        pm = p.copy(); pm[ia] -= h
        H[a, a] = (f(pp) - 2.0 * f0 + f(pm)) / (h * h)
        for b in range(a + 1, k):
            ib = idx[b]
            ppp = p.copy(); ppp[ia] += h; ppp[ib] += h
            ppm = p.copy(); ppm[ia] += h; ppm[ib] -= h
            pmp = p.copy(); pmp[ia] -= h; pmp[ib] += h
            pmm = p.copy(); pmm[ia] -= h; pmm[ib] -= h
            H[a, b] = H[b, a] = (f(ppp) - f(ppm) - f(pmp) + f(pmm)) / (4.0 * h * h)
    try:
        cov = np.linalg.inv(H)
        d = np.diag(cov)
        if np.any(d <= 0) or not np.all(np.isfinite(d)):
            return np.nan, np.nan
        return float(np.sqrt(d[0])), float(np.sqrt(d[1]))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Build a true-null simulated panel and refit it unrestricted per asset.
# Returns per-asset deltas + audit-only SEs and d_bar, or None only if an asset
# remains failed/degenerate after the same one-time rescue used by c7/c9.
# ---------------------------------------------------------------------------
def _simulate_null_returns(rng):
    """
    Simulate one true-null return panel from the fitted NULL DGP:
      eps_{j,t} = innov_{j,t} * sqrt(sigma2_null_{j,t}),  returns = mean + eps.
    innovations: Student-t copula (per-asset fitted nu margins, R_z dependence,
    shared chi-square tail-mixing) on the union calendar. Unavailable asset
    coordinates are discarded. Byte-identical innovation builder to c9's null
    draw.
    """
    obs = _G["observed"]
    nu_vec = _G["nu_null"]
    nu_c = _G["nu_c_null"]
    z_by_asset = _student_t_calendar_innovations(
        rng, nu_vec, nu_c, state=_G
    )
    returns_by_asset = {}
    for a in ASSETS:
        sig2 = obs[a]["sigma2_null"]
        mean_r = obs[a]["mean_return"]
        eps = z_by_asset[a] * np.sqrt(sig2)
        returns_by_asset[a] = mean_r + eps
    return returns_by_asset


def _panel_estimate(args):
    """
    One simulated true-null panel: refit unrestricted per asset, return the 6
    per-asset (delta_infra, delta_reg, se_infra, se_reg), d_bar, and an
    se_ok flag. A panel is ACCEPTED (kept for ALL rungs) iff every asset's refit
    converges and is non-degenerate (|delta|<=DELTA_CAP) after at most one
    deterministic six-start rescue, exactly matching the reference-draw
    optimiser effort and refit screen. Numerical-Hessian SEs are a separate,
    audit-only diagnostic and never gate a panel or a decision. Returns a tuple
    or None (None only on residual convergence/degeneracy failure).
    """
    seed = args
    rng = np.random.default_rng(seed)
    design = _G["design"]
    returns_by_asset = _simulate_null_returns(rng)
    di = np.empty(len(ASSETS)); dr = np.empty(len(ASSETS))
    sei = np.empty(len(ASSETS)); ser = np.empty(len(ASSETS))
    se_ok = True
    for k, a in enumerate(ASSETS):
        est = FastTARCHX(returns_by_asset[a], design[a]["exog_unr"])
        p, f, ok = est.fit(start=None, max_iter=MAX_ITER)
        if not ok or not c7._plausible_delta(p):
            p, f, ok = est.fit_multistart(
                n_starts=6, seed=_G.get("rescue_seed", 20260807), max_iter=MAX_ITER
            )
            if not ok or not c7._plausible_delta(p):
                return None
        di[k], dr[k] = p[5], p[6]
        # Per-asset model SEs (numerical Hessian) are AUDIT-ONLY: neither the
        # naive iid rung (Welch on the 6 vs 6 coefficients) nor the design-effect
        # rung (dispersion of the 6 paired diffs, c6's rule) uses them, so an SE
        # failure never drops a panel or changes a decision. Computed for the
        # report only; se_ok flags whether all 6 sub-blocks were PD.
        si, sr = _delta_ses(est, p)
        if not (np.isfinite(si) and np.isfinite(sr)):
            se_ok = False
            sei[k] = ser[k] = np.nan
        else:
            sei[k], ser[k] = si, sr
    d_bar = float((di - dr).mean())
    return (di, dr, sei, ser, d_bar, se_ok)


def _draw_dbar_null_gaussian(args):
    """Gaussian-copula null d_bar draw (the c7 reference distribution)."""
    return c7._draw_parametric_null(args)


def _draw_dbar_null_tcopula(args):
    """Student-t-copula null d_bar draw (the c9 reference distribution)."""
    # reuse c9's union-calendar Student-t innovation path
    seed = args
    rng = np.random.default_rng(seed)
    returns_by_asset = _simulate_null_returns(rng)
    return c7._refit_unrestricted_dbar(returns_by_asset)


# ---------------------------------------------------------------------------
# Inference-ladder decisions on one panel's recovered coefficients.
# ---------------------------------------------------------------------------
def _decisions(di, dr, sei, ser, d_bar, se_ok, rho_return, crit):
    """
    Returns a dict of {method: {alpha: reject_bool}} for one panel.

    All four rungs use a ONE-SIDED (upper-tail) test of H0: no differential
    effect vs H1: infra > reg -- the directional asymmetry the paper claims, and
    exactly the rule c7/c9 apply (reject if d_bar exceeds the upper-tail quantile
    of the null distribution). One-sided is the apples-to-apples comparison: the
    bootstrap rungs are intrinsically one-sided as the paper uses them, so the
    t-tests are made one-sided too. crit holds the one-sided upper-tail d_bar
    quantiles for the bootstrap rungs (alpha=0.05 -> 95th pct, alpha=0.10 ->
    90th pct of the null reference distribution).

    The (i) naive iid test does NOT use the per-asset model SEs (Welch on the 6
    vs 6 coefficients), so it is always defined. The (ii) design-effect rung uses
    the dispersion of the 6 paired differences (also no model SE), so it too is
    always defined. se_ok therefore does not gate any rung in the current ladder;
    it is tracked only so the per-asset model-SE pathway can be audited.
    """
    N = len(di)
    out = {}

    # (i) naive iid Welch t-test on the 6 vs 6 coefficients, ONE-SIDED (infra>reg)
    t_i, p_two_i = stats.ttest_ind(di, dr, equal_var=False)
    p_i = (p_two_i / 2.0) if t_i > 0 else (1.0 - p_two_i / 2.0)
    out["naive_iid"] = {0.05: p_i < 0.05, 0.10: p_i < 0.10, "p": p_i}

    # (ii) design-effect corrected (c6 rule): paired difference, SE inflated by
    #      sqrt(1+(N-1)rho_bar) using the cross-asset RETURN correlation. One-sided.
    d = di - dr
    mean_d = d.mean()
    se_naive = d.std(ddof=1) / np.sqrt(N)
    deff = np.sqrt(max(1.0 + (N - 1) * rho_return, 1e-12))
    se_de = se_naive * deff
    if se_de <= 0 or not np.isfinite(se_de):
        out["design_effect"] = {0.05: False, 0.10: False, "p": np.nan}
    else:
        t_ii = mean_d / se_de
        p_ii = stats.t.sf(t_ii, df=N - 1)          # one-sided upper tail
        out["design_effect"] = {0.05: p_ii < 0.05, 0.10: p_ii < 0.10, "p": p_ii}

    # (iii)/(iv) bootstrap rungs: reject if the panel's d_bar exceeds the
    #      ONE-SIDED upper-tail critical value of the respective null sampling
    #      distribution -- precisely the c7/c9 decision (p = P(null >= d_bar) < a).
    for m, key in (("gaussian_copula_boot", "gauss"), ("tcopula_boot", "t")):
        out[m] = {
            0.05: d_bar > crit[key][0.05],
            0.10: d_bar > crit[key][0.10],
            "d_bar": d_bar,
        }
    return out


# ---------------------------------------------------------------------------
def run_reference_distribution_full(draw_fn, B_ref, n_jobs, base_seed):
    """Run one seed-indexed reference law and retain exclusions as NaNs."""
    if B_ref <= 0 or n_jobs <= 0:
        raise ValueError("B_ref and n_jobs must be positive")
    seeds = np.arange(base_seed, base_seed + B_ref, dtype=np.int64)
    with Pool(processes=n_jobs) as pool:
        values = pool.map(
            draw_fn,
            seeds.tolist(),
            chunksize=max(1, B_ref // (n_jobs * 4)),
        )
    full = np.asarray(values, dtype=float)
    if full.shape != (B_ref,):
        raise RuntimeError(
            f"reference draw shape {full.shape} does not match requested {(B_ref,)}"
        )
    usable = ~np.isnan(full)
    if not np.any(usable):
        raise RuntimeError("all reference draws were excluded")
    return {"seeds": seeds, "full": full, "usable_mask": usable}


def run_reference_distributions_full(B_ref, n_jobs, base_seed):
    """Run both reference laws with explicit, disjoint deterministic streams."""
    gaussian = run_reference_distribution_full(
        _draw_dbar_null_gaussian, B_ref, n_jobs, base_seed
    )
    student_t = run_reference_distribution_full(
        _draw_dbar_null_tcopula,
        B_ref,
        n_jobs,
        base_seed + T_REFERENCE_SEED_OFFSET,
    )
    return {"gaussian": gaussian, "student_t": student_t}


def run_reference_distributions(B_ref, n_jobs, base_seed):
    """
    Calibrate the Gaussian-copula and Student-t-copula NULL sampling distributions
    of d_bar (the bootstrap critical values for rungs iii/iv). These use the same
    draw mechanisms as c7 (Gaussian) and c9 (t-copula), on c10's own documented
    reference seed streams.  This wrapper preserves the historical return API.
    """
    result = run_reference_distributions_full(B_ref, n_jobs, base_seed)
    g_full = result["gaussian"]["full"]
    t_full = result["student_t"]["full"]
    g_mask = result["gaussian"]["usable_mask"]
    t_mask = result["student_t"]["usable_mask"]
    return (
        g_full[g_mask],
        t_full[t_mask],
        int((~g_mask).sum()),
        int((~t_mask).sum()),
    )


def evaluate_panel_results(panel_results, panel_seeds, rho_return, crit):
    """Summarise seed-indexed panel results without losing failed positions."""
    panel_seeds = np.asarray(panel_seeds, dtype=np.int64)
    if len(panel_results) != len(panel_seeds):
        raise RuntimeError("panel result and seed counts differ")
    n_requested = len(panel_seeds)
    n_assets = len(ASSETS)
    usable = np.zeros(n_requested, dtype=bool)
    se_ok_full = np.zeros(n_requested, dtype=bool)
    di_full = np.full((n_requested, n_assets), np.nan, dtype=float)
    dr_full = np.full((n_requested, n_assets), np.nan, dtype=float)
    sei_full = np.full((n_requested, n_assets), np.nan, dtype=float)
    ser_full = np.full((n_requested, n_assets), np.nan, dtype=float)
    dbar_full = np.full(n_requested, np.nan, dtype=float)
    p_iid_full = np.full(n_requested, np.nan, dtype=float)
    p_de_full = np.full(n_requested, np.nan, dtype=float)
    methods = ["naive_iid", "design_effect", "gaussian_copula_boot", "tcopula_boot"]
    counts = {m: {0.05: 0, 0.10: 0} for m in methods}

    for i, panel in enumerate(panel_results):
        if panel is None:
            continue
        di, dr, sei, ser, d_bar, se_ok = panel
        di = np.asarray(di, dtype=float)
        dr = np.asarray(dr, dtype=float)
        sei = np.asarray(sei, dtype=float)
        ser = np.asarray(ser, dtype=float)
        if any(
            value.shape != (n_assets,) for value in (di, dr, sei, ser)
        ):
            raise RuntimeError(f"panel {i} has malformed per-asset coefficients")
        dec = _decisions(di, dr, sei, ser, d_bar, se_ok, rho_return, crit)
        usable[i] = True
        se_ok_full[i] = bool(se_ok)
        di_full[i] = di
        dr_full[i] = dr
        sei_full[i] = sei
        ser_full[i] = ser
        dbar_full[i] = float(d_bar)
        p_iid_full[i] = float(dec["naive_iid"]["p"])
        p_de_full[i] = float(dec["design_effect"]["p"])
        for method in methods:
            counts[method][0.05] += int(dec[method][0.05])
            counts[method][0.10] += int(dec[method][0.10])

    if not np.any(usable):
        raise RuntimeError("all outer panels were excluded")
    return {
        "panel_seeds": panel_seeds,
        "panel_usable_mask": usable,
        "panel_se_ok_full": se_ok_full,
        "delta_infra_full": di_full,
        "delta_reg_full": dr_full,
        "se_infra_full": sei_full,
        "se_reg_full": ser_full,
        "dbar_full": dbar_full,
        "p_iid_full": p_iid_full,
        "p_de_full": p_de_full,
        "counts": counts,
    }


def crit_from_draws(draws):
    """
    ONE-SIDED upper-tail critical d_bar values, exactly as c7/c9 use them: the
    null-imposed bootstrap rejects H0 (no differential effect) in favour of
    infra>reg when the observed d_bar exceeds the upper-tail quantile of the null
    sampling distribution. So the critical value at level alpha is the (1-alpha)
    quantile of the raw null draws (which already carry the estimator's small
    finite-sample location bias, since the null-imposed draws are centred at the
    biased-under-equality location, not at 0). alpha=0.05 -> 95th pct;
    alpha=0.10 -> 90th pct. centre/median reported for diagnostics only.
    """
    return {
        0.05: float(np.percentile(draws, 95)),   # one-sided 5%
        0.10: float(np.percentile(draws, 90)),   # one-sided 10%
        "centre": float(np.median(draws)),
        "mean": float(np.mean(draws)),
    }


def current_c10_analysis_provenance(c9_analysis_stack_sha256):
    """Return a deterministic contract for c10's downstream decision logic.

    The c9 fingerprints identify the fitted design, DGP, simulator, and common
    refitter used here.  They do not identify c10's outer-panel construction,
    numerical-SE diagnostic, critical-value rule, or inference-ladder
    decisions.  This contract binds those additional load-bearing choices to
    their current source and then chains them to the exact parent c9 stack.
    """
    parent = str(c9_analysis_stack_sha256)
    constants = (
        f"N_STARTS_FIT={N_STARTS_FIT}\n"
        f"MAX_ITER={MAX_ITER}\n"
        f"DELTA_CAP={DELTA_CAP:.17g}\n"
        f"FIT_SEED_DEFAULT={FIT_SEED_DEFAULT}\n"
        f"MC_SEED_DEFAULT={MC_SEED_DEFAULT}\n"
        f"RESCUE_SEED={RESCUE_SEED}\n"
        f"REFERENCE_SEED_OFFSET={REFERENCE_SEED_OFFSET}\n"
        f"T_REFERENCE_SEED_OFFSET={T_REFERENCE_SEED_OFFSET}\n"
        f"C10_SEED_SCHEMA_VERSION={C10_SEED_SCHEMA_VERSION}\n"
        f"ASSETS={','.join(ASSETS)}"
    )
    functions = (
        setup_fitted_dgp,
        _delta_ses,
        _simulate_null_returns,
        _panel_estimate,
        _draw_dbar_null_gaussian,
        _draw_dbar_null_tcopula,
        _decisions,
        run_reference_distribution_full,
        run_reference_distributions_full,
        evaluate_panel_results,
        crit_from_draws,
        main,
    )
    contract = (
        f"c10-analysis-contract-v{C10_ANALYSIS_CONTRACT_VERSION}\n"
        + constants
        + "\n"
        + "\n".join(inspect.getsource(function) for function in functions)
    )
    analysis_sha256 = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    source_sha256 = _sha256_file(Path(__file__).resolve())
    stack_sha256 = hashlib.sha256(
        (
            f"c10-downstream-analysis-stack-v{C10_ANALYSIS_CONTRACT_VERSION}\n"
            f"{parent}\n{analysis_sha256}\n{source_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "c10_analysis_contract_version": C10_ANALYSIS_CONTRACT_VERSION,
        "c10_analysis_contract": contract,
        "c10_analysis_sha256": analysis_sha256,
        "c10_source_sha256": source_sha256,
        "c10_parent_c9_analysis_stack_sha256": parent,
        "c10_downstream_analysis_stack_sha256": stack_sha256,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000, help="number of simulated null panels")
    ap.add_argument("--B-ref", "--B_ref", dest="B_ref", type=int, default=4000,
                    help="reference draws to calibrate bootstrap critical values")
    ap.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=22)
    ap.add_argument("--fit-seed", type=int, default=FIT_SEED_DEFAULT,
                    help="observed-data multistart seed; must match c9 baseline")
    ap.add_argument("--mc-seed", type=int, default=MC_SEED_DEFAULT,
                    help="outer/reference Monte-Carlo base seed")
    args = ap.parse_args()
    if args.N <= 0 or args.B_ref <= 0 or args.n_jobs <= 0:
        ap.error("N, B_ref, and n_jobs must be positive")

    t_start = time.time()
    print(f"numba available: {_HAVE_NUMBA}")
    print("Building design and fitting observed NULL + unrestricted DGP...")
    t0 = time.time()
    setup = setup_fitted_dgp(args.fit_seed, require_c9=True)
    c10_provenance = current_c10_analysis_provenance(
        setup["provenance"]["audit_analysis_stack_sha256"]
    )
    print(f"  done {time.time()-t0:.1f}s; exact c9 schema-v2 identity PASS")
    print(f"  events: {len(setup['inf_d'])} infra, {len(setup['reg_d'])} reg")
    d_bar_obs = setup["d_bar_obs"]
    multiplier = setup["multiplier"]
    rho_return = setup["rho_return"]
    rho_resid = setup["rho_resid"]
    nu_null = setup["nu_null"]
    nu_c_null = setup["nu_c_null"]
    print(f"  d_bar_obs={d_bar_obs:.4f}  multiplier={multiplier:.3f}x  (point estimate; the SIM imposes the NULL)")
    print(f"  rho_return={rho_return:.4f}  rho_resid={rho_resid:.4f}")
    print(f"  fitted nu (null) = {np.round(nu_null,3)}  median nu_c={nu_c_null:.3f}")

    # ---- (A) calibrate bootstrap critical values from reference null draws ----
    print(f"\n[A] Calibrating bootstrap critical values, B_ref={args.B_ref}...")
    t0 = time.time()
    reference_base_seed = args.mc_seed + REFERENCE_SEED_OFFSET
    reference = run_reference_distributions_full(
        args.B_ref, args.n_jobs, reference_base_seed
    )
    g_full = reference["gaussian"]["full"]
    t_full = reference["student_t"]["full"]
    g_mask = reference["gaussian"]["usable_mask"]
    t_mask = reference["student_t"]["usable_mask"]
    g_draws = g_full[g_mask]
    t_draws = t_full[t_mask]
    g_drop = int((~g_mask).sum())
    t_drop = int((~t_mask).sum())
    print(f"  done {time.time()-t0:.1f}s  "
          f"gaussian used={len(g_draws)} dropped={g_drop} ({g_drop/args.B_ref:.1%})  "
          f"t-copula used={len(t_draws)} dropped={t_drop} ({t_drop/args.B_ref:.1%})")
    crit = {"gauss": crit_from_draws(g_draws), "t": crit_from_draws(t_draws)}
    print(f"  Gaussian upper-tail d_bar critical: 5%={crit['gauss'][0.05]:.4f}  10%={crit['gauss'][0.10]:.4f}  (centre {crit['gauss']['centre']:.4f})")
    print(f"  t-copula upper-tail d_bar critical: 5%={crit['t'][0.05]:.4f}  10%={crit['t'][0.10]:.4f}  (centre {crit['t']['centre']:.4f})")

    # ---- (B) outer Monte-Carlo: simulate N true-null panels, refit, decide ----
    print(f"\n[B] Monte-Carlo size loop, N={args.N} true-null panels...")
    t0 = time.time()
    panel_seeds = np.arange(args.mc_seed, args.mc_seed + args.N, dtype=np.int64)
    with Pool(processes=args.n_jobs) as pool:
        panel_results = pool.map(_panel_estimate, panel_seeds.tolist(),
                          chunksize=max(1, args.N // (args.n_jobs * 4)))
    panel_audit = evaluate_panel_results(
        panel_results, panel_seeds, rho_return, crit
    )
    usable_panels = panel_audit["panel_usable_mask"]
    n_used = int(usable_panels.sum())
    n_fail = int((~usable_panels).sum())
    print(f"  done {time.time()-t0:.1f}s  panels used={n_used}  failed/degenerate={n_fail} ({n_fail/args.N:.1%})")

    # tally rejections
    methods = ["naive_iid", "design_effect", "gaussian_copula_boot", "tcopula_boot"]
    counts = panel_audit["counts"]
    p_iid = panel_audit["p_iid_full"][usable_panels]
    p_de = panel_audit["p_de_full"][usable_panels]
    dbar_panels = panel_audit["dbar_full"][usable_panels]
    se_pd_frac = float(
        panel_audit["panel_se_ok_full"][usable_panels].mean()
    )

    def size_and_se(c, n):
        s = c / n
        se = np.sqrt(s * (1 - s) / n)   # Monte-Carlo binomial SE
        return s, se

    print("\n=== EMPIRICAL SIZE (rejection rate under a TRUE NULL) ===")
    print(f"{'method':<24} {'size@0.05':>14} {'size@0.10':>14}")
    rows = []
    for m in methods:
        s5, se5 = size_and_se(counts[m][0.05], n_used)
        s10, se10 = size_and_se(counts[m][0.10], n_used)
        print(f"{m:<24} {s5:>8.3f}+/-{se5:.3f} {s10:>8.3f}+/-{se10:.3f}")
        rows.append({
            "method": m,
            "size_005": s5, "se_005": se5,
            "size_010": s10, "se_010": se10,
            "rej_005": counts[m][0.05], "rej_010": counts[m][0.10],
            "n_panels_used": n_used,
        })
    df = pd.DataFrame(rows)

    # over-rejection factors vs nominal
    naive5 = df.loc[df.method == "naive_iid", "size_005"].iloc[0]
    naive10 = df.loc[df.method == "naive_iid", "size_010"].iloc[0]
    t5 = df.loc[df.method == "tcopula_boot", "size_005"].iloc[0]
    t10 = df.loc[df.method == "tcopula_boot", "size_010"].iloc[0]
    print(f"\nNaive over-rejection: {naive5/0.05:.1f}x nominal at 0.05, {naive10/0.10:.1f}x at 0.10.")
    print(
        f"t-copula size (internal self-consistency check): {t5:.3f} @0.05, "
        f"{t10:.3f} @0.10 (target 0.05/0.10)."
    )

    c10_provenance_end = current_c10_analysis_provenance(
        setup["provenance"]["audit_analysis_stack_sha256"]
    )
    if c10_provenance_end != c10_provenance:
        raise RuntimeError(
            "c10 source/analysis contract changed during computation; rerun cleanly"
        )

    # ---- save ----
    out_csv = c2.OUT_DIR / "c10-size-study-results.csv"
    meta = pd.DataFrame([{
        "N_requested": args.N, "N_used": n_used, "panel_fail_frac": n_fail / args.N,
        "se_pd_frac": se_pd_frac,
        "B_ref": args.B_ref, "B_ref_used_gauss": len(g_draws), "B_ref_used_t": len(t_draws),
        "B_ref_drop_gauss": g_drop / args.B_ref, "B_ref_drop_t": t_drop / args.B_ref,
        "rho_return": rho_return, "rho_resid": rho_resid,
        "n_calendar": int(_G["n_calendar"]),
        "availability_pattern_codes": ";".join(
            str(int(v)) for v in _G["availability_pattern_codes"]
        ),
        "availability_pattern_counts": ";".join(
            str(int(v)) for v in _G["availability_pattern_counts"]
        ),
        "nu_c_null": nu_c_null, "nu_null": ";".join(f"{x:.3f}" for x in nu_null),
        "d_bar_obs_point": d_bar_obs, "multiplier_point": multiplier,
        "crit_gauss_005": crit["gauss"][0.05], "crit_gauss_010": crit["gauss"][0.10],
        "crit_t_005": crit["t"][0.05], "crit_t_010": crit["t"][0.10],
        "naive_overrej_005": naive5 / 0.05, "naive_overrej_010": naive10 / 0.10,
        "seed": args.mc_seed,  # legacy alias retained for old readers
        "fit_seed": args.fit_seed, "mc_seed": args.mc_seed,
        "rescue_seed": RESCUE_SEED,
        "panel_first_seed": int(panel_seeds[0]),
        "panel_last_seed": int(panel_seeds[-1]),
        "gaussian_reference_first_seed": int(reference["gaussian"]["seeds"][0]),
        "gaussian_reference_last_seed": int(reference["gaussian"]["seeds"][-1]),
        "t_reference_first_seed": int(reference["student_t"]["seeds"][0]),
        "t_reference_last_seed": int(reference["student_t"]["seeds"][-1]),
        "c9_seed_schema_version": setup["provenance"]["c9_seed_schema_version"],
        "design_sha256": setup["provenance"]["audit_design_sha256"],
        "fixed_null_dgp_sha256": setup["provenance"]["audit_fixed_null_dgp_sha256"],
        "refitter_sha256": setup["provenance"]["audit_refitter_sha256"],
        "simulator_sha256": setup["provenance"]["audit_simulator_sha256"],
        "analysis_stack_sha256": setup["provenance"]["audit_analysis_stack_sha256"],
        "c10_analysis_contract_version": c10_provenance[
            "c10_analysis_contract_version"
        ],
        "c10_analysis_sha256": c10_provenance["c10_analysis_sha256"],
        "c10_source_sha256": c10_provenance["c10_source_sha256"],
        "c10_parent_c9_analysis_stack_sha256": c10_provenance[
            "c10_parent_c9_analysis_stack_sha256"
        ],
        "c10_downstream_analysis_stack_sha256": c10_provenance[
            "c10_downstream_analysis_stack_sha256"
        ],
        "numba": _HAVE_NUMBA,
    }])
    metadata_csv = c2.OUT_DIR / "c10-size-study-metadata.csv"
    out_csv_tmp = out_csv.with_name(out_csv.stem + ".write-tmp.csv")
    metadata_csv_tmp = metadata_csv.with_name(
        metadata_csv.stem + ".write-tmp.csv"
    )
    df.to_csv(out_csv_tmp, index=False)
    meta.to_csv(metadata_csv_tmp, index=False)

    audit = setup["audit_payload"]
    draws_path = c2.OUT_DIR / "c10-size-study-draws.npz"
    draws_tmp = draws_path.with_name(draws_path.stem + ".write-tmp.npz")
    np.savez(
        draws_tmp,
        # Legacy compressed keys retained.
        dbar_panels=dbar_panels,
        p_iid=p_iid,
        p_de=p_de,
        g_draws=g_draws,
        t_draws=t_draws,
        crit_gauss=np.array([crit["gauss"][0.05], crit["gauss"][0.10]]),
        crit_t=np.array([crit["t"][0.05], crit["t"][0.10]]),
        # Seed-indexed outer panels.
        panel_seeds=panel_audit["panel_seeds"],
        panel_usable_mask=usable_panels,
        panel_excluded_mask=~usable_panels,
        panel_se_ok_full=panel_audit["panel_se_ok_full"],
        panel_delta_infra_full=panel_audit["delta_infra_full"],
        panel_delta_reg_full=panel_audit["delta_reg_full"],
        panel_se_infra_full=panel_audit["se_infra_full"],
        panel_se_reg_full=panel_audit["se_reg_full"],
        panel_dbar_full=panel_audit["dbar_full"],
        panel_p_iid_full=panel_audit["p_iid_full"],
        panel_p_de_full=panel_audit["p_de_full"],
        # Seed-indexed reference draws.
        gaussian_reference_seeds=reference["gaussian"]["seeds"],
        gaussian_reference_full=g_full,
        gaussian_reference_usable_mask=g_mask,
        gaussian_reference_excluded_mask=~g_mask,
        t_reference_seeds=reference["student_t"]["seeds"],
        t_reference_full=t_full,
        t_reference_usable_mask=t_mask,
        t_reference_excluded_mask=~t_mask,
        # Explicit seed contract.
        c10_seed_schema_version=np.int64(C10_SEED_SCHEMA_VERSION),
        c10_seed_schema=np.asarray(
            "fit_seed defines observed multistart only; panel_seeds[i]=mc_seed+i; "
            "gaussian reference starts mc_seed+700000; t reference adds 500000; "
            "full arrays preserve seed order and use NaN for excluded draws"
        ),
        fit_seed=np.int64(args.fit_seed),
        mc_seed=np.int64(args.mc_seed),
        seed=np.int64(args.mc_seed),
        rescue_seed=np.int64(RESCUE_SEED),
        panel_first_seed=np.int64(panel_seeds[0]),
        panel_last_seed=np.int64(panel_seeds[-1]),
        gaussian_reference_first_seed=np.int64(reference["gaussian"]["seeds"][0]),
        gaussian_reference_last_seed=np.int64(reference["gaussian"]["seeds"][-1]),
        t_reference_first_seed=np.int64(reference["student_t"]["seeds"][0]),
        t_reference_last_seed=np.int64(reference["student_t"]["seeds"][-1]),
        reference_seed_offset=np.int64(REFERENCE_SEED_OFFSET),
        t_reference_seed_offset=np.int64(T_REFERENCE_SEED_OFFSET),
        N_requested=np.int64(args.N),
        B_ref_requested=np.int64(args.B_ref),
        # DGP, calendar, and exact c9 identity.
        asset_names=np.asarray(ASSETS),
        null_params_by_asset=audit["audit_null_params_by_asset"],
        null_mean_returns=audit["audit_null_mean_returns"],
        null_sigma2_concat=audit["audit_null_sigma2_concat"],
        null_sigma2_offsets=audit["audit_null_sigma2_offsets"],
        nu_null_by_asset=audit["audit_nu_null_by_asset"],
        nu_c_null=audit["audit_nu_c_null"],
        R_z=audit["audit_R_z"],
        L_z=audit["audit_L_z"],
        simulation_calendar_ns=audit["audit_simulation_calendar_ns"],
        calendar_pos_concat=audit["audit_calendar_pos_concat"],
        calendar_pos_offsets=audit["audit_calendar_pos_offsets"],
        availability_matrix=audit["audit_availability_matrix"],
        availability_pattern_code=audit["audit_availability_pattern_code"],
        availability_pattern_codes=audit["audit_availability_pattern_codes"],
        availability_pattern_counts=audit["audit_availability_pattern_counts"],
        availability_pattern_first_date_ns=audit[
            "audit_availability_pattern_first_date_ns"
        ],
        availability_pattern_last_date_ns=audit[
            "audit_availability_pattern_last_date_ns"
        ],
        c9_seed_schema_version=np.int64(
            setup["provenance"]["c9_seed_schema_version"]
        ),
        audit_design_sha256=audit["audit_design_sha256"],
        audit_fixed_null_dgp_sha256=audit["audit_fixed_null_dgp_sha256"],
        audit_refitter_sha256=audit["audit_refitter_sha256"],
        audit_simulator_sha256=audit["audit_simulator_sha256"],
        audit_analysis_stack_sha256=audit["audit_analysis_stack_sha256"],
        c9_exact_identity_verified=np.bool_(True),
        # C10-specific current-source and downstream-decision contract.  The
        # parent c9 hash above remains unchanged for backward compatibility.
        c10_analysis_contract_version=np.int64(
            c10_provenance["c10_analysis_contract_version"]
        ),
        c10_analysis_contract=np.asarray(
            c10_provenance["c10_analysis_contract"]
        ),
        c10_analysis_sha256=np.asarray(c10_provenance["c10_analysis_sha256"]),
        c10_source_sha256=np.asarray(c10_provenance["c10_source_sha256"]),
        c10_parent_c9_analysis_stack_sha256=np.asarray(
            c10_provenance["c10_parent_c9_analysis_stack_sha256"]
        ),
        c10_downstream_analysis_stack_sha256=np.asarray(
            c10_provenance["c10_downstream_analysis_stack_sha256"]
        ),
    )
    with np.load(draws_tmp, allow_pickle=False) as trial:
        if trial["panel_dbar_full"].shape != (args.N,):
            raise RuntimeError("c10 temporary archive lost the full panel shape")
        if trial["gaussian_reference_full"].shape != (args.B_ref,) or \
                trial["t_reference_full"].shape != (args.B_ref,):
            raise RuntimeError("c10 temporary archive lost a reference-draw shape")
        if not np.array_equal(trial["panel_seeds"], panel_seeds):
            raise RuntimeError("c10 temporary archive changed panel seed order")
        if not np.array_equal(
            trial["panel_usable_mask"], ~np.isnan(trial["panel_dbar_full"])
        ):
            raise RuntimeError("c10 temporary archive panel mask/value mismatch")
        for prefix in ("gaussian", "t"):
            if not np.array_equal(
                trial[f"{prefix}_reference_usable_mask"],
                ~np.isnan(trial[f"{prefix}_reference_full"]),
            ):
                raise RuntimeError(
                    f"c10 temporary archive {prefix} reference mask/value mismatch"
                )
        for compressed, full_key, mask_key in (
            ("dbar_panels", "panel_dbar_full", "panel_usable_mask"),
            ("p_iid", "panel_p_iid_full", "panel_usable_mask"),
            ("p_de", "panel_p_de_full", "panel_usable_mask"),
            ("g_draws", "gaussian_reference_full", "gaussian_reference_usable_mask"),
            ("t_draws", "t_reference_full", "t_reference_usable_mask"),
        ):
            if not np.array_equal(
                trial[compressed], trial[full_key][trial[mask_key]], equal_nan=True
            ):
                raise RuntimeError(
                    f"c10 temporary archive compressed/full mismatch for {compressed}"
                )
        if not bool(trial["c9_exact_identity_verified"]):
            raise RuntimeError("c10 temporary archive lost the c9 identity flag")
        for key in (
            "c10_analysis_contract",
            "c10_analysis_sha256",
            "c10_source_sha256",
            "c10_parent_c9_analysis_stack_sha256",
            "c10_downstream_analysis_stack_sha256",
        ):
            if _scalar_text(trial[key]) != str(c10_provenance[key]):
                raise RuntimeError(f"c10 temporary archive changed {key}")
        if int(trial["c10_analysis_contract_version"]) != int(
            c10_provenance["c10_analysis_contract_version"]
        ):
            raise RuntimeError(
                "c10 temporary archive changed the analysis-contract version"
            )
    meta_trial = pd.read_csv(metadata_csv_tmp)
    if len(meta_trial) != 1:
        raise RuntimeError("c10 metadata did not round-trip as one row")
    for key in (
        "c10_analysis_sha256",
        "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    ):
        if str(meta_trial.loc[0, key]) != str(c10_provenance[key]):
            raise RuntimeError(f"c10 metadata changed {key}")
    draws_tmp.replace(draws_path)
    out_csv_tmp.replace(out_csv)
    metadata_csv_tmp.replace(metadata_csv)
    print(f"\nSaved {out_csv} and {metadata_csv}")

    write_finding(df, meta.iloc[0], crit, time.time() - t_start)
    print(f"\nTOTAL {(time.time()-t_start)/60:.1f} min")


def write_finding(df, meta, crit, elapsed):
    def row(m):
        r = df.loc[df.method == m].iloc[0]
        return r

    naive = row("naive_iid"); de = row("design_effect")
    gb = row("gaussian_copula_boot"); tb = row("tcopula_boot")

    L = []
    L.append("# C10 -- Monte-Carlo Size-Distortion Study of the Inference Ladder\n")
    L.append(f"_N={int(meta['N_used'])} true-null panels (of {int(meta['N_requested'])} requested); "
             f"bootstrap critical values calibrated from B_ref={int(meta['B_ref'])}; "
             f"numba={meta['numba']}; runtime {elapsed/60:.1f} min._\n")

    L.append("## The question\n")
    L.append("How often do the inference rules reject under the fitted fixed-design null? "
             "This is an internal calibration diagnostic. It directly measures over-rejection "
             "of the naive rule in this DGP, but it does not independently validate the plug-in "
             "bootstrap because panels and critical values share the same fitted generator.\n")

    L.append("## DGP (true null, fitted to the data)\n")
    L.append(f"- Per-asset **GJR-GARCH-X** variance dynamics estimated under the "
             f"**null-imposed** combined-dummy spec (D_event = D_infra + D_reg), so "
             f"delta_infra = delta_reg **by construction** -- there is genuinely no "
             f"differential event effect in the truth.")
    L.append(f"- Cross-asset dependence at the **standardised-residual correlation** "
             f"rho_resid = {meta['rho_resid']:.3f} (return-correlation rho_return = "
             f"{meta['rho_return']:.3f}), via the Cholesky of R_z.")
    L.append(f"- **Student-t innovations** at each asset's fitted nu = "
             f"[{meta['nu_null']}] (median nu_c = {meta['nu_c_null']:.3f}), unit variance, "
             f"with joint tail dependence via a true multivariate-t copula (shared "
             f"chi-square mixing). The fit seed and schema-v2 DGP, design, refitter, "
             f"simulator, and analysis fingerprints are asserted exactly against c9 "
             f"before any Monte-Carlo draws.")
    L.append(f"- The **actual 26 infra / 24 reg** event-label structure and the real "
             f"event-window dummies are reused unchanged for every panel; only returns "
             f"are re-simulated. (Point estimate on the real data: d_bar = "
             f"{meta['d_bar_obs_point']:.3f}, {meta['multiplier_point']:.2f}x -- the SIM "
             f"imposes the null, this is shown only to anchor the DGP.)\n")

    L.append("## The size table (empirical rejection rate under a true null)\n")
    L.append("| method | size @ alpha=0.05 | size @ alpha=0.10 |")
    L.append("|---|---|---|")
    L.append(f"| (i) NAIVE iid t-test | **{naive['size_005']:.3f}** +/- {naive['se_005']:.3f} | "
             f"**{naive['size_010']:.3f}** +/- {naive['se_010']:.3f} |")
    L.append(f"| (ii) design-effect corrected | {de['size_005']:.3f} +/- {de['se_005']:.3f} | "
             f"{de['size_010']:.3f} +/- {de['se_010']:.3f} |")
    L.append(f"| (iii) Gaussian-copula bootstrap | {gb['size_005']:.3f} +/- {gb['se_005']:.3f} | "
             f"{gb['size_010']:.3f} +/- {gb['se_010']:.3f} |")
    L.append(f"| (iv) Student-t-copula bootstrap | {tb['size_005']:.3f} +/- {tb['se_005']:.3f} | "
             f"{tb['size_010']:.3f} +/- {tb['se_010']:.3f} |")
    L.append(f"\n_(+/- = Monte-Carlo binomial standard error.) Nominal size is 0.05 and 0.10._\n")

    L.append("## Internal self-consistency check\n")
    z5 = ((tb['size_005'] - 0.05) / tb['se_005']
          if tb['se_005'] > 0 else float("nan"))
    z10 = ((tb['size_010'] - 0.10) / tb['se_010']
           if tb['se_010'] > 0 else float("nan"))
    L.append(f"The Student-t-copula rule is calibrated from the same fixed DGP used to draw "
             f"the panels, so near-nominal rejection is an expected implementation check, "
             f"not independent validation. It comes out "
             f"**{tb['size_005']:.3f} @0.05** and **{tb['size_010']:.3f} @0.10** "
             f"(targets 0.05/0.10). Relative to the reported binomial Monte Carlo "
             f"standard errors, the deviations are **{z5:+.2f} SE** and "
             f"**{z10:+.2f} SE**, respectively.\n")

    L.append("## Verdict on the naive over-rejection\n")
    over5 = float(meta['naive_overrej_005']); over10 = float(meta['naive_overrej_010'])
    L.append(f"- The **naive iid t-test rejects a true null {naive['size_005']:.1%} of the "
             f"time at the 5% level ({over5:.1f}x nominal) and {naive['size_010']:.1%} at "
             f"the 10% level ({over10:.1f}x nominal).** This is severe size distortion: a "
             f"'significant' headline from this rule is, under the fitted null, a "
             f"false positive a large fraction of the time. It is the direct, "
             f"simulated counterpart of the pseudoreplication diagnosis -- the six assets "
             f"are not six independent draws.")
    L.append(
        f"- The **design-effect t(5) rule** rejects at "
        f"{de['size_005']:.3f}/{de['size_010']:.3f}, versus "
        f"{naive['size_005']:.3f}/{naive['size_010']:.3f} for the naive rule. "
        "It therefore does not reduce over-rejection in this run. The rule uses "
        "raw-return correlation as a proxy and does not reproduce the fitted "
        "GARCH/heavy-tail structure."
    )
    L.append(f"- The **Gaussian-copula bootstrap** lands at "
             f"{gb['size_005']:.3f}/{gb['size_010']:.3f} and the **Student-t-copula "
             f"bootstrap** at {tb['size_005']:.3f}/{tb['size_010']:.3f} -- "
             f"under this fitted DGP. The load-bearing result is the severe over-rejection "
             f"of the naive and Gaussian rules; the t-copula result is internally calibrated.\n")

    L.append("## Method notes\n")
    L.append("- (i) NAIVE: Welch t-test on the 6 per-asset delta_infra vs 6 delta_reg as "
             "iid (the headline rule that produced t=4.768/p=0.0008).")
    L.append("- (ii) DESIGN-EFFECT: paired-difference SE inflated by "
             "sqrt(1+(N-1)*rho_bar) on the cross-asset RETURN correlation, t on N-1 df "
             "(c6's rule).")
    L.append("- (iii)/(iv) BOOTSTRAP: ONE-SIDED upper-tail test (H1: infra>reg) -- reject if "
             "d_bar exceeds the (1-alpha) quantile of the respective null sampling "
             "distribution of d_bar (Gaussian copula = c7; Student-t copula = c9), exactly "
             "the c7/c9 decision p=P(null>=d_bar)<alpha. Critical values are calibrated once "
             "from B_ref reference draws under the same fixed DGP. This tests self-consistency; "
             "a fully nested re-estimation would be required for independent plug-in validation. "
             f"One-sided critical d_bar: Gaussian "
             f"5%={crit['gauss'][0.05]:.4f}/10%={crit['gauss'][0.10]:.4f}, "
             f"t-copula 5%={crit['t'][0.05]:.4f}/10%={crit['t'][0.10]:.4f} "
             f"(null centres {crit['gauss']['centre']:.4f} / {crit['t']['centre']:.4f}).\n")

    L.append("## Fragility / honest caveats\n")
    L.append(f"- **Convergence:** {meta['panel_fail_frac']:.1%} of the {int(meta['N_requested'])} "
             f"outer panels remained excluded after the shared deterministic rescue "
             f"(a per-asset refit failed to converge or hit a degenerate "
             f"|delta|>{DELTA_CAP:.0f} optimum). Reference-draw drop rates: Gaussian "
             f"{meta['B_ref_drop_gauss']:.1%}, t-copula {meta['B_ref_drop_t']:.1%}. All under "
             f"the 10% reliability threshold "
             + ("(reliable)." if (meta['panel_fail_frac'] < 0.10 and
                                   meta['B_ref_drop_gauss'] < 0.10 and
                                   meta['B_ref_drop_t'] < 0.10) else
                "-- EXCEEDED somewhere; treat the affected column with caution.") + "")
    L.append(f"- **Per-asset model SEs are audit-only, not on the decision path.** Neither "
             f"rung (i) (Welch on the 6 vs 6 coefficients) nor rung (ii) (dispersion of the "
             f"6 paired differences, c6's rule) uses the per-asset model SE, so an SE "
             f"failure never drops a panel or flips a decision. The numerical-Hessian SEs "
             f"are computed only for reporting; their sub-block was positive-definite for "
             f"{meta['se_pd_frac']:.1%} of accepted panels. Panel acceptance is therefore "
             f"identical to the reference-draw rule (convergence + |delta|<={DELTA_CAP:.0f}), "
             f"and refit screen matches the reference-draw rule.")
    L.append("- **The bootstrap rungs are calibrated, not fully nested.** Rungs (iii)/(iv) "
             "measure self-consistency under one shared fitted fixed-path DGP. A fully "
             "nested design that re-estimates the null DGP and critical values for every "
             "panel would be required for independent plug-in validation.")
    L.append("- **Copula df** for the t-copula tail-mixing is the median fitted nu (one "
             "shared dependence parameter), with fitted per-asset Student-t margins. The "
             "Gaussian comparator retains the same fitted R_z linear-correlation input but "
             "uses Gaussian margins and has no shared-mixing tail dependence. Their contrast "
             "therefore changes the full joint innovation law---both margins and tail "
             "dependence---rather than isolating a margins-only effect; neither law is "
             "guaranteed correct under the unknown true DGP.")
    L.append("- **Monte-Carlo error**: sizes carry binomial SEs of ~"
             f"{naive['se_005']:.3f}-{naive['se_010']:.3f}; differences within ~2 SE of "
             "each other or of nominal are not separable at this N.\n")

    L.append("## Files\n")
    L.append("- `c10-size-study-results.csv` -- method-by-level size table")
    L.append("- `c10-size-study-metadata.csv` -- DGP, screening, and calibration metadata")
    L.append("- `c10-size-study-draws.npz` -- full seed-indexed panel coefficients/SEs, "
             "per-panel p-values (iid/deff), reference null draws, masks, critical values, "
             "and exact DGP/calendar provenance")
    L.append("- `code/c10_size_study.py` (reuses `tarch_x_fast.py`, "
             "`c7_ccc_garchx_bootstrap.py`, `c9_tcopula_bootstrap.py`)")

    (c2.OUT_DIR / "c10-size-study-FINDING.md").write_text("\n".join(L))
    print(f"Saved {c2.OUT_DIR / 'c10-size-study-FINDING.md'}")


if __name__ == "__main__":
    main()
