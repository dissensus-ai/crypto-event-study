"""
c19: Recursive-bootstrap sensitivity for the second-moment null.

PURPOSE
-------
Compare recursive volatility propagation with c9's committed conditional
fixed-path bootstrap for the baseline or asset-specific high-variance
specification (legacy machine key ``crisis``). C19 uses the
same observed-fit seed (12345), the same draw seeds (22345 through 24344), the
same Student-t-copula innovation construction, and the same unrestricted-refit
function as c9.  The innovation draws are therefore paired by seed; the sole
simulation-design difference is whether the fitted null variance path is held
fixed (c9) or regenerated through the GJR recursion (c19).

PRESPECIFIED DESIGN (fixed before running; nothing tuned on results)
--------------------------------------------------------------------
* B = 2000, seeds = BASE_SEED + 10_000 + i with BASE_SEED = 12345, identical
  to c9's committed run.
* Null DGP: per-asset NULL-imposed fit (combined event dummy), exactly the
  c7.fit_observed output c9 uses. Innovations: c9._student_t_innovations
  (multivariate-t copula at R_z, per-asset fitted nu margins, unit variance).
  One full six-vector and one shared chi-square mixer are drawn per union-
  calendar date; unavailable coordinates are discarded -- verbatim c9
  construction, including the five-asset marginal on pre-BNB dates.
* RECURSION (the one deliberate difference from c9): instead of scaling the
  drawn z against the FIXED fitted path sigma2_null, propagate
      sig2_t = omega + alpha*eps_{t-1}^2 + gamma*eps_{t-1}^2*1(eps<0)
               + beta*sig2_{t-1} + x_t' delta_null,
  floored at 1e-8, initialised at sigma2_null[0]; eps_t = sqrt(sig2_t)*z_t,
  r_t = mean_return + eps_t. Exogenous regressors (event dummies, sentiment)
  stay at their actual values, as in every scheme.
* FAILURE HANDLING: c19 uses c7/c9's convergence and degeneracy screen
  (|delta| > DELTA_CAP = 50), including exactly one deterministic six-start
  rescue after any failed single-start asset refit.  The opt-in diagnostics
  from c7 count both draw-level and asset-refit-level rescue use.  Recursive
  simulation additionally needs a path guard: a draw whose recursion exceeds
  SIG2_EXPLODE = 1e10 at any t is flagged "exploded" and excluded before
  refitting; explosion exclusions are reported separately. Reported p-values:
    p_one           = (#{d* >= d_obs} + 1) / (B_used + 1)     [c9 convention]
    p_abs           = (#{|d*| >= |d_obs|} + 1) / (B_used + 1)
    p_equal_tail    = min(1, 2 * min(upper, lower)) with the same smoothing
    p_worst_low     = (#{d* >= d_obs} + 1) / (B + 1)          [failures -> below]
    p_worst_high    = (#{d* >= d_obs} + n_total_exclusions + 1) / (B + 1)
                      [all exclusions -> at/above]
  The worst-case bounds are the honest envelope: with a materially selective
  failure rate the point p is not trustworthy, which is itself a finding.
* DIAGNOSTICS (the adjudication payload):
  - null mean/SD of d* under the recursive scheme vs the fixed scheme
    (fixed distribution loaded from c9-tcopula-draws-baseline.npz);
  - per-draw realized return variance (mean across assets) vs the ACTUAL
    per-asset sample variances -- the fitted null persistence is ~0.999, so
    the implied unconditional variance omega/(1-persist) is far above the
    sample variance; a recursive simulation wanders toward that level while
    the observed sample never does. If realized variances are inflated /
    wildly dispersed relative to the data, the recursive null is simulating
    a different market than the one observed;
  - rescue and exclusion accounting: draws requiring the shared six-start
    rescue, draws retained after it, residual refit exclusions, and recursion-
    explosion exclusions are recorded under unambiguous names.
  - positivity-guard accounting: the number of draws, asset paths, and asset-days
    on which the pre-floor recursive variance falls below 1e-8, the minimum raw
    variance, per-asset counts, and the recursive-null distribution split by
    floor-hit status.  This is distinct from the fitted-path floor diagnostic.

Outputs
  results/c19-recursive-sensitivity-{SPEC}.csv
  results/c19-recursive-draws-{SPEC}.npz

For backward compatibility, a baseline run also refreshes
``results/c19-recursive-sensitivity.csv`` and
``results/c19-recursive-draws.npz`` as explicit byte-for-byte baseline aliases.
The crisis run never overwrites those aliases. Once both current-schema
specifications exist, the script regenerates the combined narrative FINDING.

Telemetry-only replay (same simulation path, no model refits):
  python c19_recursive_sensitivity.py baseline --telemetry-only
  python c19_recursive_sensitivity.py crisis --telemetry-only

Fixed-reference sync after a c9 rerun (no bootstrap simulation):
  python c19_recursive_sensitivity.py baseline --sync-fixed-only
  python c19_recursive_sensitivity.py crisis --sync-fixed-only
This mode reconstructs the deterministic observed fit to validate c9's DGP and
refitter fingerprint, validates the complete seed schema and exact observed
statistic, then replaces only fixed-reference fields in the existing c19
artifacts. All recursive draws and telemetry are checked unchanged before commit.
"""
import argparse
import csv
import hashlib
import inspect
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_relaxed_threshold_sensitivity as c2
import c7_ccc_garchx_bootstrap as c7
import c8h_break_controls_ccc_bootstrap as c8h
import c9_tcopula_bootstrap as c9
from c7_ccc_garchx_bootstrap import ASSETS
from multiprocessing import Pool

SPEC = "baseline"
TELEMETRY_ONLY = False
SYNC_FIXED_ONLY = False
B = 2000
BASE_SEED = 12345             # c9 committed run; draw seeds = 22_345 + i
DRAW_SEED_OFFSET = 10_000
RECURSIVE_SEED_SCHEMA_VERSION = 1
N_JOBS = 22
SIG2_EXPLODE = 1e10
SIG2_FLOOR = 1e-8
P_VALUE_CONTRACT = (
    "retained-draw add-one smoothing: p_one=(upper_hits+1)/(B_used+1); "
    "p_abs=(absolute_hits+1)/(B_used+1); "
    "p_equal=min(1,2*min((upper_hits+1)/(B_used+1),"
    "(lower_hits+1)/(B_used+1)))"
)


def _parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed B=2000 floored-recursive sensitivity for one c9 "
            "specification."
        )
    )
    parser.add_argument(
        "spec",
        nargs="?",
        default="baseline",
        choices=("baseline", "crisis"),
        help="model specification to regenerate (default: baseline)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--telemetry-only",
        action="store_true",
        help="replay recursive paths and refresh floor telemetry without refitting",
    )
    modes.add_argument(
        "--sync-fixed-only",
        action="store_true",
        help="validate existing recursive draws and refresh only the c9 fixed reference",
    )
    return parser.parse_args(argv)


def _simulate_recursive_panel(seed):
    """Generate one recursive null panel and its path-level telemetry.

    This helper contains the exact innovation construction and variance recursion
    used by the inferential draw.  Keeping telemetry replay on this same path
    prevents a diagnostic-only implementation from drifting from the numerical
    algorithm whose committed draws it audits.
    """
    rng = np.random.default_rng(seed)
    obs = c7._GLOBAL["observed"]
    design = c7._GLOBAL["design"]
    nu_vec = c7._GLOBAL["nu_null"]
    nu_c = c7._GLOBAL["nu_c_null"]

    # -- identical innovation construction to c9._draw_parametric_null_t -----
    z_by_asset = c9._student_t_calendar_innovations(rng, nu_vec, nu_c)
    returns_by_asset = {}
    exploded = False
    max_sig2 = 0.0
    realized_vars = []
    variance_floor_hits_by_asset = np.zeros(len(ASSETS), dtype=np.int64)
    min_raw_variance_by_asset = np.full(len(ASSETS), np.inf, dtype=float)
    for j, a in enumerate(ASSETS):
        sig2_null = obs[a]["sigma2_null"]
        mean_r = obs[a]["mean_return"]
        T = sig2_null.shape[0]
        z_full = z_by_asset[a]

        # -- the one difference from c9: propagate the null recursion --------
        pn = obs[a]["params_null"]           # [omega, alpha, gamma, beta, nu, deltas(4)]
        omega, alpha, gamma, beta = pn[0], pn[1], pn[2], pn[3]
        exog_contrib = design[a]["exog_null"] @ pn[5:]   # T-vector
        sig2 = np.empty(T)
        eps = np.empty(T)
        # Initialise the recursion AT the fitted null variance for t=0 (rather than
        # feeding sigma2_null[0] in as sigma^2_{-1}, which applied one extra
        # recursion step before the first observation).
        sig2[0] = max(sig2_null[0], SIG2_FLOOR)
        eps[0] = np.sqrt(sig2[0]) * z_full[0]
        max_sig2 = max(max_sig2, float(sig2[0]))
        s_prev = sig2[0]
        e_prev = eps[0]
        for t in range(1, T):
            raw_s = omega + alpha * e_prev * e_prev \
                + gamma * e_prev * e_prev * (e_prev < 0.0) \
                + beta * s_prev + exog_contrib[t]
            min_raw_variance_by_asset[j] = min(
                min_raw_variance_by_asset[j], float(raw_s)
            )
            if np.isfinite(raw_s) and raw_s < SIG2_FLOOR:
                variance_floor_hits_by_asset[j] += 1
            s = raw_s
            if not np.isfinite(s) or s > SIG2_EXPLODE:
                exploded = True
                s = min(max(s, SIG2_FLOOR), SIG2_EXPLODE) if np.isfinite(s) else SIG2_EXPLODE
            s = max(s, SIG2_FLOOR)
            e = np.sqrt(s) * z_full[t]
            sig2[t] = s
            eps[t] = e
            s_prev = s
            e_prev = e
        max_sig2 = max(max_sig2, float(sig2.max()))
        r = mean_r + eps
        realized_vars.append(float(np.var(r)))
        returns_by_asset[a] = r

    mean_rv = float(np.mean(realized_vars))
    n_variance_floor_hits = int(variance_floor_hits_by_asset.sum())
    n_asset_paths_with_variance_floor_hit = int(
        np.count_nonzero(variance_floor_hits_by_asset)
    )
    min_raw_variance = float(np.min(min_raw_variance_by_asset))
    telemetry = (
        float(exploded),
        max_sig2,
        mean_rv,
        float(n_variance_floor_hits),
        float(n_asset_paths_with_variance_floor_hit),
        min_raw_variance,
        *variance_floor_hits_by_asset.astype(float),
        *min_raw_variance_by_asset,
    )
    return returns_by_asset, telemetry


def _draw_recursive_telemetry(seed):
    """Replay one paired seed through simulation only (no model refits)."""
    _returns_by_asset, telemetry = _simulate_recursive_panel(seed)
    return telemetry


def _draw_recursive_null_t(args):
    """Run one paired-seed recursive draw and return value plus audit flags.

    The returned tuple contains dbar, recursion/path diagnostics, draw-level
    rescue-attempt and rescue-retention flags, asset-refit-level rescue counts,
    and variance-floor telemetry.  Adding telemetry does not alter the simulated
    panel or the refit path.
    """
    returns_by_asset, telemetry = _simulate_recursive_panel(args)
    exploded = telemetry[0] > 0.5
    if exploded:
        return (np.nan, *telemetry[:3], 0.0, 0.0, 0.0, 0.0, *telemetry[3:])

    # This is the same refitter c9 calls.  Its default scalar interface remains
    # unchanged; c19 opts into telemetry for the shared single rescue path.
    dbar, refit_diag = c7._refit_unrestricted_dbar(
        returns_by_asset, return_diagnostics=True
    )
    return (
        dbar,
        0.0,
        telemetry[1],
        telemetry[2],
        float(refit_diag["draw_had_six_start_rescue_attempt"]),
        float(refit_diag["draw_retained_after_six_start_rescue"]),
        float(refit_diag["n_asset_refits_with_six_start_rescue_attempt"]),
        float(refit_diag["n_asset_refits_rescued_by_six_start"]),
        *telemetry[3:],
    )


def _recursive_audit_payload(fixed_audit):
    """Fingerprint the code and seed contract unique to the recursive scheme.

    The shared c9 audit identifies the data, fitted null DGP, refitter, and
    union-calendar innovation engine.  This additional digest closes the
    provenance gap for c19's own recursion, variance floor/explosion guards,
    and inferential wrapper.
    """
    contract = (
        f"recursive_seed_schema_version={RECURSIVE_SEED_SCHEMA_VERSION};"
        f"draw_seed_offset={DRAW_SEED_OFFSET};"
        f"SIG2_FLOOR={SIG2_FLOOR:.17g};"
        f"SIG2_EXPLODE={SIG2_EXPLODE:.17g};"
        f"p_value_contract={P_VALUE_CONTRACT}\n"
        + inspect.getsource(_simulate_recursive_panel)
        + inspect.getsource(_draw_recursive_null_t)
    )
    simulator_sha256 = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    fixed_stack_sha256 = str(fixed_audit["audit_analysis_stack_sha256"])
    analysis_stack_sha256 = hashlib.sha256(
        (SPEC + "\n" + fixed_stack_sha256 + "\n" + simulator_sha256).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "recursive_seed_schema_version": np.int64(
            RECURSIVE_SEED_SCHEMA_VERSION
        ),
        "recursive_seed_schema": np.asarray(
            "draw_seeds[i] = base_seed + recursive_draw_seed_offset + i; "
            "each seed uses the c9 union-calendar innovation vector and then "
            "propagates the c19 floored recursive variance path"
        ),
        "recursive_draw_seed_offset": np.int64(DRAW_SEED_OFFSET),
        "recursive_simulator_sha256": np.asarray(simulator_sha256),
        "recursive_analysis_stack_sha256": np.asarray(analysis_stack_sha256),
        "recursive_simulator_contract": np.asarray(contract),
    }


def _variance_floor_summary(dbar, d_bar_obs, floor_hits, floor_asset_paths,
                            min_raw, floor_hits_by_asset, min_raw_by_asset):
    """Aggregate draw- and asset-level floor telemetry for the release CSV."""
    usable = np.isfinite(dbar)
    floor_draw = floor_hits > 0
    usable_floor = usable & floor_draw
    usable_no_floor = usable & (~floor_draw)

    def subset_stats(mask, suffix):
        values = dbar[mask]
        if len(values) == 0:
            return {
                f"n_usable_draws_{suffix}": 0,
                f"n_upper_tail_hits_recursive_{suffix}": 0,
                f"n_abs_stat_hits_recursive_{suffix}": 0,
                f"null_mean_recursive_{suffix}": np.nan,
                f"null_sd_recursive_{suffix}": np.nan,
            }
        return {
            f"n_usable_draws_{suffix}": int(len(values)),
            f"n_upper_tail_hits_recursive_{suffix}": int(
                np.sum(values >= d_bar_obs)
            ),
            f"n_abs_stat_hits_recursive_{suffix}": int(
                np.sum(np.abs(values) >= abs(d_bar_obs))
            ),
            f"null_mean_recursive_{suffix}": float(values.mean()),
            f"null_sd_recursive_{suffix}": float(values.std()),
        }

    total_hits_by_asset = floor_hits_by_asset.sum(axis=0).astype(int)
    draws_hit_by_asset = (floor_hits_by_asset > 0).sum(axis=0).astype(int)
    min_by_asset = min_raw_by_asset.min(axis=0)
    summary = {
        "n_draws_with_variance_floor_hit": int(floor_draw.sum()),
        "fraction_draws_with_variance_floor_hit": float(floor_draw.mean()),
        "n_variance_floor_hits": int(floor_hits.sum()),
        "n_asset_paths_with_variance_floor_hit": int(floor_asset_paths.sum()),
        "min_raw_variance_before_floor": float(min_raw.min()),
        "variance_floor_hits_by_asset": ";".join(
            f"{a}:{n}" for a, n in zip(ASSETS, total_hits_by_asset)
        ),
        "draws_with_variance_floor_hit_by_asset": ";".join(
            f"{a}:{n}" for a, n in zip(ASSETS, draws_hit_by_asset)
        ),
        "min_raw_variance_by_asset": ";".join(
            f"{a}:{v:.16g}" for a, v in zip(ASSETS, min_by_asset)
        ),
    }
    summary.update(subset_stats(usable_floor, "with_variance_floor_hit"))
    summary.update(subset_stats(usable_no_floor, "without_variance_floor_hit"))
    return summary


def _atomic_savez(path, payload):
    """Replace an NPZ only after the complete temporary archive is written."""
    tmp = path.with_name(path.stem + ".telemetry-tmp.npz")
    np.savez(tmp, **payload)
    with np.load(tmp, allow_pickle=False) as trial:
        if set(trial.files) != set(payload):
            raise RuntimeError(f"temporary {path.name} archive lost fields")
        for key, value in payload.items():
            if not _arrays_equal_exact(trial[key], value):
                raise RuntimeError(
                    f"temporary {path.name} field {key!r} failed exact round trip"
                )
    tmp.replace(path)


def _arrays_equal_exact(left, right):
    """Exact array equality, treating NaNs at the same positions as equal."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.number):
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _reconstruct_fixed_audit():
    """Rebuild only the observed fit needed to fingerprint c19's DGP contract."""
    if SPEC == "baseline":
        design, _inf_d, _reg_d, ret_df = c7.build_design()
    elif SPEC == "crisis":
        design, _inf_d, _reg_d, ret_df, _cw = c8h.build_design_with_regimes("crisis")
    else:
        raise ValueError(SPEC)
    observed = c7.fit_observed(design, seed=BASE_SEED)
    d_bar_obs = float(np.mean([
        observed[a]["delta_infra"] - observed[a]["delta_reg"] for a in ASSETS
    ]))
    z_df = pd.DataFrame({
        a: pd.Series(observed[a]["z_resid"], index=design[a]["index"])
        for a in ASSETS
    }).dropna()
    R_z = z_df.corr().values
    R_z_pd = R_z.copy()
    eps_jit = 0.0
    while True:
        try:
            L_z = np.linalg.cholesky(R_z_pd)
            break
        except np.linalg.LinAlgError:
            eps_jit = max(eps_jit * 10, 1e-8)
            R_z_pd = R_z + eps_jit * np.eye(len(ASSETS))
    common_idx = z_df.index
    common_pos = {
        a: pd.Index(design[a]["index"]).get_indexer(common_idx) for a in ASSETS
    }
    c7._GLOBAL.update({
        "design": design,
        "observed": observed,
        "R_z": R_z,
        "L_z": L_z,
        "n_common": len(common_idx),
        "common_pos": common_pos,
        "max_len": max(observed[a]["resid_unr"].shape[0] for a in ASSETS),
    })
    c7.install_simulation_calendar(design)
    c9._install_nu(observed)
    return d_bar_obs, c9._fixed_dgp_audit_payload(SPEC)


def _sync_fixed_reference():
    """Refresh c19's embedded c9 reference without touching recursive fields."""
    if SPEC not in {"baseline", "crisis"}:
        raise ValueError("--sync-fixed-only supports baseline or crisis")

    fixed_path = c2.OUT_DIR / f"c9-tcopula-draws-{SPEC}.npz"
    draws_path = c2.OUT_DIR / f"c19-recursive-draws-{SPEC}.npz"
    csv_path = c2.OUT_DIR / f"c19-recursive-sensitivity-{SPEC}.csv"
    for path in (fixed_path, draws_path, csv_path):
        if not path.exists():
            raise FileNotFoundError(f"fixed-reference sync requires {path}")

    required_c9 = {
        "null_t", "null_t_full", "null_t_usable_mask",
        "null_t_excluded_mask", "draw_seeds", "seed_schema_version",
        "base_seed", "draw_seed_offset", "first_draw_seed",
        "last_draw_seed", "B_requested", "d_bar_obs",
        "draw_had_six_start_rescue_attempt_t",
        "draw_retained_after_six_start_rescue_t",
        "n_asset_refits_with_six_start_rescue_attempt_t",
        "n_asset_refits_rescued_by_six_start_t",
        "audit_design_sha256", "audit_fixed_null_dgp_sha256",
        "audit_refitter_sha256", "audit_simulator_sha256",
        "audit_analysis_stack_sha256",
    }
    with np.load(fixed_path) as archive:
        missing = required_c9.difference(archive.files)
        if missing:
            raise RuntimeError(
                f"{fixed_path.name} lacks full c9 audit schema: {sorted(missing)}"
            )
        fixed = {key: archive[key] for key in archive.files}

    requested = int(fixed["B_requested"])
    base_seed = int(fixed["base_seed"])
    seed_offset = int(fixed["draw_seed_offset"])
    draw_seeds = np.asarray(fixed["draw_seeds"], dtype=np.int64)
    expected_seeds = np.arange(
        base_seed + seed_offset,
        base_seed + seed_offset + requested,
        dtype=np.int64,
    )
    if int(fixed["seed_schema_version"]) != 2:
        raise RuntimeError("unsupported c9 seed_schema_version")
    if not np.array_equal(draw_seeds, expected_seeds):
        raise RuntimeError("c9 draw_seeds do not match base_seed + offset + position")
    if (int(fixed["first_draw_seed"]), int(fixed["last_draw_seed"])) != (
        int(expected_seeds[0]), int(expected_seeds[-1])
    ):
        raise RuntimeError("c9 first/last draw-seed fields do not match draw_seeds")

    null_full = np.asarray(fixed["null_t_full"], dtype=float)
    usable = np.asarray(fixed["null_t_usable_mask"], dtype=bool)
    excluded = np.asarray(fixed["null_t_excluded_mask"], dtype=bool)
    null_fix = np.asarray(fixed["null_t"], dtype=float)
    if null_full.shape != (requested,) or usable.shape != (requested,):
        raise RuntimeError("c9 full fixed draws or usable mask have the wrong length")
    if excluded.shape != (requested,) or not np.array_equal(excluded, ~usable):
        raise RuntimeError("c9 fixed usable/excluded masks are not complements")
    if not np.array_equal(usable, ~np.isnan(null_full)):
        raise RuntimeError("c9 fixed usable mask does not identify the non-NaN draws")
    if not np.array_equal(null_fix, null_full[usable]):
        raise RuntimeError("c9 legacy null_t is not the seed-ordered usable compression")

    rescue_attempted = np.asarray(
        fixed["draw_had_six_start_rescue_attempt_t"], dtype=bool
    )
    rescue_retained = np.asarray(
        fixed["draw_retained_after_six_start_rescue_t"], dtype=bool
    )
    asset_attempts = np.asarray(
        fixed["n_asset_refits_with_six_start_rescue_attempt_t"], dtype=np.int64
    )
    asset_successes = np.asarray(
        fixed["n_asset_refits_rescued_by_six_start_t"], dtype=np.int64
    )
    for name, values in {
        "rescue_attempted": rescue_attempted,
        "rescue_retained": rescue_retained,
        "asset_attempts": asset_attempts,
        "asset_successes": asset_successes,
    }.items():
        if values.shape != (requested,):
            raise RuntimeError(f"c9 {name} telemetry has the wrong length")
    if not np.array_equal(rescue_retained, rescue_attempted & usable):
        raise RuntimeError("c9 rescue-retention flags disagree with rescue and usable masks")
    if np.any(asset_successes > asset_attempts):
        raise RuntimeError("c9 rescued-asset counts exceed rescue attempts")

    with np.load(draws_path) as archive:
        original_payload = {key: archive[key] for key in archive.files}
    required_recursive = {
        "dbar", "d_bar_obs", "draw_seeds", "recursive_usable_mask",
        "recursive_excluded_mask", "B_requested", "base_seed",
        "first_draw_seed", "last_draw_seed", "recursive_seed_schema_version",
        "recursive_seed_schema", "recursive_draw_seed_offset",
        "recursive_simulator_sha256", "recursive_analysis_stack_sha256",
        "recursive_simulator_contract",
    }
    missing_recursive = required_recursive.difference(original_payload)
    if missing_recursive:
        raise RuntimeError(
            f"{draws_path.name} lacks recursive provenance fields: "
            f"{sorted(missing_recursive)}; rerun full c19"
        )
    # A fixed-only sync is valid only within the same RNG/DGP schema.  In
    # particular, never attach union-calendar c9 draws to recursive draws made
    # by the superseded common-window/independent-pre-BNB engine.
    for key in (
        "audit_design_sha256",
        "audit_fixed_null_dgp_sha256",
        "audit_refitter_sha256",
        "audit_simulator_sha256",
        "audit_analysis_stack_sha256",
    ):
        if key not in original_payload:
            raise RuntimeError(
                f"{draws_path.name} has no {key}; rerun full c19 under the current "
                "union-calendar engine before using --sync-fixed-only"
            )
        if str(original_payload[key]) != str(fixed[key]):
            raise RuntimeError(
                f"c19 recursive {key} differs from c9; full c19 rerun required"
            )
    recursive_full = np.asarray(original_payload["dbar"], dtype=float)
    recursive_usable = np.asarray(
        original_payload["recursive_usable_mask"], dtype=bool
    )
    recursive_excluded = np.asarray(
        original_payload["recursive_excluded_mask"], dtype=bool
    )
    recursive_seeds = np.asarray(original_payload["draw_seeds"], dtype=np.int64)
    recursive_B = len(recursive_full)
    if recursive_B != requested:
        raise RuntimeError(
            f"c9/c19 requested-draw mismatch: fixed={requested}, recursive={recursive_B}"
        )
    if int(original_payload["B_requested"]) != requested:
        raise RuntimeError("c19 B_requested does not match c9")
    if recursive_usable.shape != (requested,) or recursive_excluded.shape != (
        requested,
    ):
        raise RuntimeError("c19 recursive masks have the wrong length")
    if not np.array_equal(recursive_excluded, ~recursive_usable):
        raise RuntimeError("c19 recursive usable/excluded masks are not complements")
    if not np.array_equal(recursive_usable, ~np.isnan(recursive_full)):
        raise RuntimeError("c19 recursive usable mask does not identify non-NaN draws")
    if int(original_payload["recursive_seed_schema_version"]) != (
        RECURSIVE_SEED_SCHEMA_VERSION
    ) or int(original_payload["recursive_draw_seed_offset"]) != seed_offset:
        raise RuntimeError("c19 recursive seed schema differs from c9 pairing contract")
    if not np.array_equal(recursive_seeds, expected_seeds):
        raise RuntimeError("c19 recursive draw_seeds differ from c9 paired seeds")
    for key, expected in (
        ("base_seed", base_seed),
        ("first_draw_seed", int(expected_seeds[0])),
        ("last_draw_seed", int(expected_seeds[-1])),
    ):
        if key not in original_payload or int(original_payload[key]) != expected:
            raise RuntimeError(f"c19 {key} does not match c9's exact seed schema")
    fixed_d_bar_obs = float(fixed["d_bar_obs"])
    recursive_d_bar_obs = float(original_payload["d_bar_obs"])
    if recursive_d_bar_obs != fixed_d_bar_obs:
        raise RuntimeError(
            "c9/c19 d_bar_obs differs exactly: "
            f"fixed={fixed_d_bar_obs:.17g}, recursive={recursive_d_bar_obs:.17g}"
        )

    print("  reconstructing observed fit for DGP/refitter fingerprint assertion...")
    local_d_bar_obs, local_audit = _reconstruct_fixed_audit()
    if local_d_bar_obs != fixed_d_bar_obs:
        raise RuntimeError(
            "locally reconstructed c19 d_bar_obs differs exactly from c9: "
            f"local={local_d_bar_obs:.17g}, fixed={fixed_d_bar_obs:.17g}"
        )
    for key in (
        "audit_design_sha256",
        "audit_fixed_null_dgp_sha256",
        "audit_refitter_sha256",
        "audit_simulator_sha256",
        "audit_analysis_stack_sha256",
    ):
        if str(local_audit[key]) != str(fixed[key]):
            raise RuntimeError(f"locally reconstructed c19 {key} differs from c9")
    local_recursive_audit = _recursive_audit_payload(local_audit)
    for key in (
        "recursive_seed_schema_version",
        "recursive_seed_schema",
        "recursive_draw_seed_offset",
        "recursive_simulator_sha256",
        "recursive_analysis_stack_sha256",
        "recursive_simulator_contract",
    ):
        if not _arrays_equal_exact(original_payload[key], local_recursive_audit[key]):
            raise RuntimeError(
                f"committed c19 {key} differs from the local recursive implementation"
            )

    fixed_updates = {
        "null_fixed": null_fix,
        "null_fixed_full": null_full,
        "null_fixed_usable_mask": usable,
        "null_fixed_excluded_mask": excluded,
        "fixed_draw_seeds": draw_seeds,
        "fixed_seed_schema_version": np.int64(2),
        "fixed_base_seed": np.int64(base_seed),
        "fixed_draw_seed_offset": np.int64(seed_offset),
        "fixed_first_draw_seed": np.int64(expected_seeds[0]),
        "fixed_last_draw_seed": np.int64(expected_seeds[-1]),
        "fixed_draw_had_six_start_rescue_attempt": rescue_attempted,
        "fixed_draw_retained_after_six_start_rescue": rescue_retained,
        "fixed_n_asset_refits_with_six_start_rescue_attempt": asset_attempts,
        "fixed_n_asset_refits_rescued_by_six_start": asset_successes,
        "fixed_audit_design_sha256": np.asarray(str(fixed["audit_design_sha256"])),
        "fixed_audit_fixed_null_dgp_sha256": np.asarray(
            str(fixed["audit_fixed_null_dgp_sha256"])
        ),
        "fixed_audit_refitter_sha256": np.asarray(
            str(fixed["audit_refitter_sha256"])
        ),
        "fixed_audit_simulator_sha256": np.asarray(
            str(fixed["audit_simulator_sha256"])
        ),
        "fixed_audit_analysis_stack_sha256": np.asarray(
            str(fixed["audit_analysis_stack_sha256"])
        ),
        "fixed_audit_locally_reconstructed_equal": np.bool_(True),
    }
    updated_payload = dict(original_payload)
    updated_payload.update(fixed_updates)
    tmp_npz = draws_path.with_name(draws_path.stem + ".sync-fixed-tmp.npz")
    np.savez(tmp_npz, **updated_payload)
    with np.load(tmp_npz) as archive:
        trial_payload = {key: archive[key] for key in archive.files}
    for key, value in original_payload.items():
        if key in fixed_updates:
            continue
        if key not in trial_payload or not _arrays_equal_exact(value, trial_payload[key]):
            raise RuntimeError(f"sync would alter recursive NPZ field {key!r}")
    for key, value in fixed_updates.items():
        if key not in trial_payload or not _arrays_equal_exact(value, trial_payload[key]):
            raise RuntimeError(f"fixed NPZ field {key!r} failed round-trip validation")

    hits_upper = int(np.sum(null_fix >= fixed_d_bar_obs))
    hits_abs = int(np.sum(np.abs(null_fix) >= abs(fixed_d_bar_obs)))
    hits_lower = int(np.sum(null_fix <= fixed_d_bar_obs))
    B_used = len(null_fix)
    if B_used == 0:
        raise RuntimeError("c9 fixed reference has no retained draws")
    p_one = (hits_upper + 1) / (B_used + 1)
    p_abs = (hits_abs + 1) / (B_used + 1)
    p_equal = min(1.0, 2 * min(p_one, (hits_lower + 1) / (B_used + 1)))
    fixed_summary = {
        "B_used_fixed": B_used,
        "n_upper_tail_hits_fixed": hits_upper,
        "n_abs_stat_hits_fixed": hits_abs,
        "n_lower_tail_hits_fixed": hits_lower,
        "p_one_fixed": p_one,
        "p_abs_fixed": p_abs,
        "p_equal_tail_fixed": p_equal,
        "null_mean_fixed": float(null_fix.mean()),
        "null_sd_fixed": float(null_fix.std()),
        "B_requested_fixed": requested,
        "n_total_exclusions_fixed": int(excluded.sum()),
        "fraction_total_excluded_fixed": float(excluded.mean()),
        "n_draws_with_six_start_rescue_attempt_fixed": int(rescue_attempted.sum()),
        "n_draws_retained_after_six_start_rescue_fixed": int(rescue_retained.sum()),
        "n_draws_failed_after_six_start_rescue_fixed": int(
            (rescue_attempted & excluded).sum()
        ),
        "n_asset_refits_with_six_start_rescue_attempt_fixed": int(
            asset_attempts.sum()
        ),
        "n_asset_refits_rescued_by_six_start_fixed": int(asset_successes.sum()),
        "design_sha256_fixed": str(fixed["audit_design_sha256"]),
        "fixed_null_dgp_sha256_fixed": str(fixed["audit_fixed_null_dgp_sha256"]),
        "refitter_sha256_fixed": str(fixed["audit_refitter_sha256"]),
        "simulator_sha256_fixed": str(fixed["audit_simulator_sha256"]),
        "analysis_stack_sha256_fixed": str(fixed["audit_analysis_stack_sha256"]),
        "fixed_audit_locally_reconstructed_equal": True,
    }

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 1 or rows[0].get("spec") != SPEC:
        raise RuntimeError(f"unexpected committed CSV contents in {csv_path.name}")
    original_row = dict(rows[0])
    if int(original_row["B"]) != requested:
        raise RuntimeError("c19 CSV B does not match c9 requested draws")
    if int(original_row.get("B_requested", -1)) != requested:
        raise RuntimeError("c19 CSV B_requested does not match c9 requested draws")
    if int(original_row.get("recursive_seed_schema_version", -1)) != (
        RECURSIVE_SEED_SCHEMA_VERSION
    ) or int(original_row.get("recursive_draw_seed_offset", -1)) != seed_offset:
        raise RuntimeError("c19 CSV recursive seed schema is incomplete or mismatched")
    if (
        int(original_row["base_seed"]),
        int(original_row["first_draw_seed"]),
        int(original_row["last_draw_seed"]),
    ) != (base_seed, int(expected_seeds[0]), int(expected_seeds[-1])):
        raise RuntimeError("c19 CSV seed fields do not match c9's exact seed schema")
    if float(original_row["d_bar_obs"]) != fixed_d_bar_obs:
        raise RuntimeError("c19 CSV d_bar_obs does not exactly match c9")
    for csv_key, audit_key in (
        ("recursive_simulator_sha256", "recursive_simulator_sha256"),
        ("recursive_analysis_stack_sha256", "recursive_analysis_stack_sha256"),
    ):
        if original_row.get(csv_key) != str(local_recursive_audit[audit_key]):
            raise RuntimeError(f"c19 CSV {csv_key} differs from local implementation")

    updated_row = dict(original_row)
    for key, value in fixed_summary.items():
        updated_row[key] = str(value)
        if key not in fieldnames:
            fieldnames.append(key)
    tmp_csv = csv_path.with_name(csv_path.stem + ".sync-fixed-tmp.csv")
    with tmp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(updated_row)
    with tmp_csv.open(newline="") as handle:
        trial_rows = list(csv.DictReader(handle))
    if len(trial_rows) != 1:
        raise RuntimeError("fixed-summary CSV failed round-trip validation")
    trial_row = trial_rows[0]
    for key, value in original_row.items():
        if key not in fixed_summary and trial_row.get(key) != value:
            raise RuntimeError(f"sync would alter recursive CSV field {key!r}")
    for key, value in fixed_summary.items():
        if trial_row.get(key) != str(value):
            raise RuntimeError(f"fixed CSV field {key!r} failed round-trip validation")

    # Both temporary artifacts are complete and independently validated before
    # either committed artifact is replaced.
    tmp_npz.replace(draws_path)
    tmp_csv.replace(csv_path)
    if SPEC == "baseline":
        draws_alias = c2.OUT_DIR / "c19-recursive-draws.npz"
        csv_alias = c2.OUT_DIR / "c19-recursive-sensitivity.csv"
        shutil.copyfile(draws_path, draws_alias)
        shutil.copyfile(csv_path, csv_alias)
        if draws_alias.read_bytes() != draws_path.read_bytes():
            raise RuntimeError("baseline c19 NPZ alias is not byte-identical")
        if csv_alias.read_bytes() != csv_path.read_bytes():
            raise RuntimeError("baseline c19 CSV alias is not byte-identical")

    print(
        f"  synced {SPEC}: fixed retained={B_used}/{requested}, "
        f"excluded={int(excluded.sum())}, upper/abs/lower hits="
        f"{hits_upper}/{hits_abs}/{hits_lower}, p_one={p_one:.16g}"
    )
    print(
        "  fixed rescue telemetry: draws attempted/retained/failed="
        f"{int(rescue_attempted.sum())}/{int(rescue_retained.sum())}/"
        f"{int((rescue_attempted & excluded).sum())}; asset attempts/rescued="
        f"{int(asset_attempts.sum())}/{int(asset_successes.sum())}"
    )
    print("  recursive NPZ fields and recursive CSV fields validated unchanged")
    _write_finding_if_complete()
    return fixed_summary


def _replay_and_commit_floor_telemetry(d_bar_obs):
    """Replay simulation only and append exact telemetry to committed artifacts.

    Existing recursive dbar values and all inferential fields are preserved.  The
    replay is accepted only if its explosion, maximum-variance, and realised-
    variance arrays exactly match the already committed simulation diagnostics.
    """
    print(f"\n[TELEMETRY-ONLY REPLAY] B={B} n_jobs={N_JOBS}...")
    seeds = np.arange(
        BASE_SEED + DRAW_SEED_OFFSET,
        BASE_SEED + DRAW_SEED_OFFSET + B,
        dtype=np.int64,
    )
    with Pool(processes=N_JOBS) as pool:
        replay = pool.map(
            _draw_recursive_telemetry,
            seeds.tolist(),
            chunksize=max(1, B // (N_JOBS * 4)),
        )
    replay = np.asarray(replay, dtype=float)
    replay_exploded = replay[:, 0] > 0.5
    replay_max_sig2 = replay[:, 1]
    replay_mean_rv = replay[:, 2]
    floor_hits = replay[:, 3].astype(np.int64)
    floor_asset_paths = replay[:, 4].astype(np.int64)
    min_raw = replay[:, 5]
    floor_hits_by_asset = replay[:, 6:12].astype(np.int64)
    min_raw_by_asset = replay[:, 12:18]

    draws_path = c2.OUT_DIR / f"c19-recursive-draws-{SPEC}.npz"
    csv_path = c2.OUT_DIR / f"c19-recursive-sensitivity-{SPEC}.csv"
    if not draws_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            f"telemetry replay requires existing {draws_path.name} and {csv_path.name}"
        )
    with np.load(draws_path) as existing:
        payload = {key: existing[key] for key in existing.files}
    dbar = np.asarray(payload["dbar"], dtype=float)
    if dbar.shape != (B,):
        raise RuntimeError(f"unexpected committed dbar shape: {dbar.shape}")
    if not np.isclose(float(payload["d_bar_obs"]), d_bar_obs, rtol=0.0, atol=1e-12):
        raise RuntimeError("telemetry replay observed statistic does not match committed draw file")
    for key, expected in (("base_seed", BASE_SEED),
                          ("B_requested", B),
                          ("recursive_seed_schema_version",
                           RECURSIVE_SEED_SCHEMA_VERSION),
                          ("recursive_draw_seed_offset", DRAW_SEED_OFFSET),
                          ("first_draw_seed", int(seeds[0])),
                          ("last_draw_seed", int(seeds[-1]))):
        if int(payload[key]) != expected:
            raise RuntimeError(f"committed {key} does not match telemetry replay")
    if not np.array_equal(np.asarray(payload["draw_seeds"], dtype=np.int64), seeds):
        raise RuntimeError("committed recursive draw_seeds differ from telemetry replay")
    usable = np.asarray(payload["recursive_usable_mask"], dtype=bool)
    excluded = np.asarray(payload["recursive_excluded_mask"], dtype=bool)
    if not np.array_equal(usable, ~np.isnan(dbar)) or not np.array_equal(
        excluded, ~usable
    ):
        raise RuntimeError("committed recursive masks differ from dbar seed positions")
    local_recursive_audit = _recursive_audit_payload(
        c9._fixed_dgp_audit_payload(SPEC)
    )
    for key in (
        "recursive_seed_schema",
        "recursive_simulator_sha256",
        "recursive_analysis_stack_sha256",
        "recursive_simulator_contract",
    ):
        if not _arrays_equal_exact(payload[key], local_recursive_audit[key]):
            raise RuntimeError(f"committed {key} differs from telemetry replay code")
    if not np.array_equal(np.asarray(payload["exploded"], dtype=bool), replay_exploded):
        raise RuntimeError("telemetry replay explosion mask differs from committed simulation")
    if not np.array_equal(np.asarray(payload["max_sig2"]), replay_max_sig2):
        raise RuntimeError("telemetry replay max_sig2 differs from committed simulation")
    if not np.array_equal(np.asarray(payload["mean_rv"]), replay_mean_rv):
        raise RuntimeError("telemetry replay mean_rv differs from committed simulation")

    floor_draw = floor_hits > 0
    payload.update({
        "draw_had_variance_floor_hit": floor_draw,
        "n_variance_floor_hits": floor_hits,
        "n_asset_paths_with_variance_floor_hit": floor_asset_paths,
        "min_raw_variance_before_floor": min_raw,
        "variance_floor_hits_by_asset": floor_hits_by_asset,
        "min_raw_variance_by_asset": min_raw_by_asset,
        "variance_floor_asset_names": np.asarray(ASSETS),
    })
    _atomic_savez(draws_path, payload)

    summary = _variance_floor_summary(
        dbar, d_bar_obs, floor_hits, floor_asset_paths, min_raw,
        floor_hits_by_asset, min_raw_by_asset,
    )
    frame = pd.read_csv(csv_path)
    if len(frame) != 1 or str(frame.iloc[0]["spec"]) != SPEC:
        raise RuntimeError(f"unexpected committed CSV contents in {csv_path.name}")
    for key, value in summary.items():
        frame.loc[0, key] = value
    tmp_csv = csv_path.with_name(csv_path.stem + ".telemetry-tmp.csv")
    frame.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    if SPEC == "baseline":
        shutil.copyfile(draws_path, c2.OUT_DIR / "c19-recursive-draws.npz")
        shutil.copyfile(csv_path, c2.OUT_DIR / "c19-recursive-sensitivity.csv")
    print(
        "  variance floor: "
        f"{summary['n_draws_with_variance_floor_hit']}/{B} draws; "
        f"{summary['n_variance_floor_hits']} asset-days across "
        f"{summary['n_asset_paths_with_variance_floor_hit']} asset paths; "
        f"minimum raw variance {summary['min_raw_variance_before_floor']:.6g}"
    )
    print(f"  committed telemetry to {draws_path.name} and {csv_path.name}")
    _write_finding_if_complete()
    return summary


def _write_finding_if_complete():
    """Generate the two-spec digest only when both current-schema CSVs exist."""
    paths = {
        spec: c2.OUT_DIR / f"c19-recursive-sensitivity-{spec}.csv"
        for spec in ("baseline", "crisis")
    }
    draw_paths = {
        spec: c2.OUT_DIR / f"c19-recursive-draws-{spec}.npz"
        for spec in ("baseline", "crisis")
    }
    if not all(path.is_file() for path in (*paths.values(), *draw_paths.values())):
        return False
    required = {
        "spec", "B", "B_requested", "B_used", "B_used_fixed",
        "base_seed", "recursive_draw_seed_offset", "first_draw_seed",
        "last_draw_seed", "n_calendar", "availability_pattern_codes",
        "availability_pattern_counts", "analysis_stack_sha256",
        "recursive_seed_schema_version", "recursive_simulator_sha256",
        "recursive_analysis_stack_sha256", "p_one_recursive",
        "p_abs_recursive", "p_equal_tail_recursive", "p_worst_low",
        "p_worst_high", "p_one_fixed", "p_abs_fixed", "p_equal_tail_fixed",
        "null_mean_recursive", "null_sd_recursive", "null_mean_fixed",
        "null_sd_fixed", "fraction_draws_with_variance_floor_hit",
        "n_variance_floor_hits", "n_asset_paths_with_variance_floor_hit",
        "n_draws_with_six_start_rescue_attempt",
        "n_draws_retained_after_six_start_rescue",
        "n_draws_failed_after_six_start_rescue",
        "realized_var_median_recursive", "sample_var_mean_actual",
    }
    rows = {}
    archives = {}
    for spec, path in paths.items():
        frame = pd.read_csv(path)
        if len(frame) != 1 or required.difference(frame.columns):
            return False
        row = frame.iloc[0].to_dict()
        if (
            str(row["spec"]) != spec
            or int(row["B"]) != B
            or int(row["B_requested"]) != B
            or int(row["recursive_seed_schema_version"])
            != RECURSIVE_SEED_SCHEMA_VERSION
        ):
            return False
        with np.load(draw_paths[spec], allow_pickle=False) as archive:
            archive_required = {
                "dbar", "draw_seeds", "recursive_usable_mask",
                "recursive_excluded_mask", "B_requested", "base_seed",
                "d_bar_obs", "null_fixed", "null_fixed_full",
                "null_fixed_usable_mask", "null_fixed_excluded_mask",
                "draw_had_six_start_rescue_attempt",
                "draw_retained_after_six_start_rescue",
                "draw_had_variance_floor_hit", "n_variance_floor_hits",
                "n_asset_paths_with_variance_floor_hit", "mean_rv",
                "sample_vars_actual", "sample_var_mean_actual",
                "recursive_draw_seed_offset", "first_draw_seed",
                "last_draw_seed", "recursive_seed_schema_version",
                "recursive_simulator_sha256", "recursive_simulator_contract",
                "recursive_analysis_stack_sha256", "audit_analysis_stack_sha256",
                "fixed_audit_analysis_stack_sha256", "audit_simulation_calendar_ns",
                "audit_availability_pattern_codes",
                "audit_availability_pattern_counts",
            }
            if archive_required.difference(archive.files):
                return False
            saved = {key: archive[key] for key in archive_required}
        draw_seeds = np.asarray(saved["draw_seeds"], dtype=np.int64)
        dbar = np.asarray(saved["dbar"], dtype=float)
        usable = np.asarray(saved["recursive_usable_mask"], dtype=bool)
        excluded = np.asarray(saved["recursive_excluded_mask"], dtype=bool)
        expected_seeds = np.arange(
            int(row["base_seed"]) + int(row["recursive_draw_seed_offset"]),
            int(row["base_seed"]) + int(row["recursive_draw_seed_offset"]) + B,
            dtype=np.int64,
        )
        simulator_contract = str(saved["recursive_simulator_contract"])
        simulator_sha256 = hashlib.sha256(
            simulator_contract.encode("utf-8")
        ).hexdigest()
        fixed_stack_sha256 = str(saved["audit_analysis_stack_sha256"])
        recursive_stack_sha256 = hashlib.sha256(
            (spec + "\n" + fixed_stack_sha256 + "\n" + simulator_sha256).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            dbar.shape != (B,)
            or usable.shape != (B,)
            or not np.array_equal(excluded, ~usable)
            or not np.array_equal(usable, ~np.isnan(dbar))
            or not np.array_equal(draw_seeds, expected_seeds)
            or int(saved["B_requested"]) != B
            or int(saved["base_seed"]) != int(row["base_seed"])
            or int(saved["recursive_draw_seed_offset"])
            != int(row["recursive_draw_seed_offset"])
            or int(saved["first_draw_seed"]) != int(row["first_draw_seed"])
            or int(saved["last_draw_seed"]) != int(row["last_draw_seed"])
            or int(saved["recursive_seed_schema_version"])
            != RECURSIVE_SEED_SCHEMA_VERSION
            or simulator_sha256 != str(saved["recursive_simulator_sha256"])
            or simulator_sha256 != str(row["recursive_simulator_sha256"])
            or recursive_stack_sha256
            != str(saved["recursive_analysis_stack_sha256"])
            or recursive_stack_sha256
            != str(row["recursive_analysis_stack_sha256"])
            or fixed_stack_sha256 != str(saved["fixed_audit_analysis_stack_sha256"])
            or fixed_stack_sha256 != str(row["analysis_stack_sha256"])
            or len(saved["audit_simulation_calendar_ns"]) != int(row["n_calendar"])
            or ";".join(
                str(int(v)) for v in saved["audit_availability_pattern_codes"]
            ) != str(row["availability_pattern_codes"])
            or ";".join(
                str(int(v)) for v in saved["audit_availability_pattern_counts"]
            ) != str(row["availability_pattern_counts"])
        ):
            return False
        fixed_full = np.asarray(saved["null_fixed_full"], dtype=float)
        fixed_usable = np.asarray(saved["null_fixed_usable_mask"], dtype=bool)
        fixed_excluded = np.asarray(saved["null_fixed_excluded_mask"], dtype=bool)
        fixed_values = np.asarray(saved["null_fixed"], dtype=float)
        rescue_attempted = np.asarray(
            saved["draw_had_six_start_rescue_attempt"], dtype=bool
        )
        rescue_retained = np.asarray(
            saved["draw_retained_after_six_start_rescue"], dtype=bool
        )
        floor_draw = np.asarray(saved["draw_had_variance_floor_hit"], dtype=bool)
        floor_hits = np.asarray(saved["n_variance_floor_hits"], dtype=np.int64)
        floor_paths = np.asarray(
            saved["n_asset_paths_with_variance_floor_hit"], dtype=np.int64
        )
        mean_rv = np.asarray(saved["mean_rv"], dtype=float)
        d_bar_obs = float(saved["d_bar_obs"])
        if (
            fixed_full.shape != (B,)
            or fixed_usable.shape != (B,)
            or not np.array_equal(fixed_excluded, ~fixed_usable)
            or not np.array_equal(fixed_usable, ~np.isnan(fixed_full))
            or not np.array_equal(fixed_values, fixed_full[fixed_usable])
            or rescue_attempted.shape != (B,)
            or not np.array_equal(rescue_retained, rescue_attempted & usable)
            or floor_draw.shape != (B,)
            or not np.array_equal(floor_draw, floor_hits > 0)
            or mean_rv.shape != (B,)
            or np.asarray(saved["sample_vars_actual"]).shape != (len(ASSETS),)
            or not np.isclose(
                float(saved["sample_var_mean_actual"]),
                float(np.mean(saved["sample_vars_actual"])),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            return False
        recursive_values = dbar[usable]
        recursive_upper = int(np.sum(recursive_values >= d_bar_obs))
        recursive_abs = int(
            np.sum(np.abs(recursive_values) >= abs(d_bar_obs))
        )
        recursive_lower = int(np.sum(recursive_values <= d_bar_obs))
        fixed_upper = int(np.sum(fixed_values >= d_bar_obs))
        fixed_abs = int(np.sum(np.abs(fixed_values) >= abs(d_bar_obs)))
        fixed_lower = int(np.sum(fixed_values <= d_bar_obs))
        recursive_p_one = (recursive_upper + 1) / (len(recursive_values) + 1)
        recursive_p_abs = (recursive_abs + 1) / (len(recursive_values) + 1)
        recursive_p_equal = min(
            1.0,
            2 * min(
                recursive_p_one,
                (recursive_lower + 1) / (len(recursive_values) + 1),
            ),
        )
        fixed_p_one = (fixed_upper + 1) / (len(fixed_values) + 1)
        fixed_p_abs = (fixed_abs + 1) / (len(fixed_values) + 1)
        fixed_p_equal = min(
            1.0,
            2 * min(fixed_p_one, (fixed_lower + 1) / (len(fixed_values) + 1)),
        )
        integer_checks = {
            "B_used": len(recursive_values),
            "B_used_fixed": len(fixed_values),
            "n_draws_with_six_start_rescue_attempt": int(rescue_attempted.sum()),
            "n_draws_retained_after_six_start_rescue": int(rescue_retained.sum()),
            "n_draws_failed_after_six_start_rescue": int(
                (rescue_attempted & ~usable).sum()
            ),
            "n_variance_floor_hits": int(floor_hits.sum()),
            "n_asset_paths_with_variance_floor_hit": int(floor_paths.sum()),
        }
        float_checks = {
            "p_one_recursive": recursive_p_one,
            "p_abs_recursive": recursive_p_abs,
            "p_equal_tail_recursive": recursive_p_equal,
            "p_worst_low": (recursive_upper + 1) / (B + 1),
            "p_worst_high": (recursive_upper + int((~usable).sum()) + 1)
            / (B + 1),
            "p_one_fixed": fixed_p_one,
            "p_abs_fixed": fixed_p_abs,
            "p_equal_tail_fixed": fixed_p_equal,
            "null_mean_recursive": float(recursive_values.mean()),
            "null_sd_recursive": float(recursive_values.std()),
            "null_mean_fixed": float(fixed_values.mean()),
            "null_sd_fixed": float(fixed_values.std()),
            "fraction_draws_with_variance_floor_hit": float(floor_draw.mean()),
            "realized_var_median_recursive": float(np.median(mean_rv[usable])),
            "sample_var_mean_actual": float(saved["sample_var_mean_actual"]),
        }
        if any(int(row[key]) != expected for key, expected in integer_checks.items()):
            return False
        if any(
            not np.isclose(
                float(row[key]), expected, rtol=1e-14, atol=1e-15, equal_nan=True
            )
            for key, expected in float_checks.items()
        ):
            return False
        rows[spec] = row
        archives[spec] = saved

    common_csv_fields = (
        "base_seed", "recursive_draw_seed_offset", "first_draw_seed",
        "last_draw_seed", "n_calendar", "availability_pattern_codes",
        "availability_pattern_counts", "recursive_simulator_sha256",
    )
    if any(
        str(rows["baseline"][key]) != str(rows["crisis"][key])
        for key in common_csv_fields
    ) or not np.array_equal(
        archives["baseline"]["draw_seeds"], archives["crisis"]["draw_seeds"]
    ):
        return False

    lines = [
        "# C19 -- Fixed-path versus floored recursive sensitivity",
        "",
        ("Both specifications use B=2,000 paired seed positions, the same "
         "union-calendar Student-t innovation vectors, and the same unrestricted "
         "refitter. The deliberate difference is whether the fitted null variance "
         "path is held fixed or propagated recursively with the stated positivity "
         "floor and explosion guard."),
        "",
        ("The retained machine label `crisis` denotes the asset-specific "
         "high-variance window from each asset's last 2021 conditional-variance "
         "break to its first break in 2022 or later; it is not an FTX-specific "
         "control."),
        "",
        "| spec | retained fixed / recursive | one-sided p fixed / recursive | absolute-statistic p fixed / recursive | null mean fixed / recursive | null SD fixed / recursive | floor-hit draws |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in ("baseline", "crisis"):
        row = rows[spec]
        lines.append(
            f"| {spec} | {int(row['B_used_fixed'])} / {int(row['B_used'])} | "
            f"{row['p_one_fixed']:.4f} / {row['p_one_recursive']:.4f} | "
            f"{row['p_abs_fixed']:.4f} / {row['p_abs_recursive']:.4f} | "
            f"{row['null_mean_fixed']:.4f} / {row['null_mean_recursive']:.4f} | "
            f"{row['null_sd_fixed']:.4f} / {row['null_sd_recursive']:.4f} | "
            f"{row['fraction_draws_with_variance_floor_hit']:.1%} |"
        )
    lines.extend([
        "",
        "The recursive p-values use add-one smoothing over retained draws. The "
        "reported worst-case bounds assign every excluded draw below or above the "
        "observed statistic; exact bounds, rescue counts, floor telemetry, and "
        "realised-variance diagnostics are in the CSV and seed-indexed NPZ files.",
        "",
        "Floor-hit versus no-floor partitions are post hoc diagnostics. Differences "
        "between those strata do not identify a causal effect of flooring. The "
        "recursive result is therefore evidence under the implemented floored "
        "algorithm, not an unconstrained recursive GJR-GARCH-X bootstrap.",
        "",
        "Files: `c19-recursive-sensitivity-{baseline,crisis}.csv` and "
        "`c19-recursive-draws-{baseline,crisis}.npz`; baseline aliases are "
        "byte-identical to the baseline-specific files.",
        "",
    ])
    finding_path = c2.OUT_DIR / "c19-recursive-sensitivity-FINDING.md"
    finding_tmp = finding_path.with_name(finding_path.stem + ".write-tmp.md")
    finding_tmp.write_text("\n".join(lines))
    if "# C19 -- Fixed-path versus floored recursive sensitivity" not in (
        finding_tmp.read_text()
    ):
        raise RuntimeError("c19 temporary finding failed validation")
    finding_tmp.replace(finding_path)
    return True


def main():
    t_start = time.time()
    if TELEMETRY_ONLY and SYNC_FIXED_ONLY:
        raise ValueError("choose only one of --telemetry-only and --sync-fixed-only")
    if SYNC_FIXED_ONLY:
        print("=" * 72)
        print(f"c19: SYNC FIXED c9 REFERENCE ONLY (spec={SPEC})")
        print("=" * 72)
        _sync_fixed_reference()
        print(f"\nTotal {time.time()-t_start:.1f}s")
        print("C19_FIXED_REFERENCE_SYNC_DONE")
        return

    print("=" * 72)
    print("c19: RECURSIVE bootstrap sensitivity (spec=%s, B=%d)" % (SPEC, B))
    print("=" * 72)

    if SPEC == "baseline":
        design, inf_d, reg_d, ret_df = c7.build_design()
    elif SPEC == "crisis":
        design, inf_d, reg_d, ret_df, _cw = c8h.build_design_with_regimes("crisis")
    else:
        raise ValueError(SPEC)
    print("Fitting observed (unrestricted + null), multistart...")
    observed = c7.fit_observed(design, seed=BASE_SEED)
    di = np.array([observed[a]["delta_infra"] for a in ASSETS])
    dr = np.array([observed[a]["delta_reg"] for a in ASSETS])
    d_bar_obs = float((di - dr).mean())
    print(f"  d_bar_obs={d_bar_obs:.4f}  multiplier={di.mean()/dr.mean():.3f}x")

    z_df = pd.DataFrame({a: pd.Series(observed[a]["z_resid"], index=design[a]["index"])
                         for a in ASSETS}).dropna()
    R_z = z_df.corr().values
    R_z_pd = R_z.copy(); eps_jit = 0.0
    while True:
        try:
            L_z = np.linalg.cholesky(R_z_pd); break
        except np.linalg.LinAlgError:
            eps_jit = max(eps_jit * 10, 1e-8)
            R_z_pd = R_z + eps_jit * np.eye(len(ASSETS))
    common_idx = z_df.index
    common_pos = {a: pd.Index(design[a]["index"]).get_indexer(common_idx) for a in ASSETS}

    c7._GLOBAL["design"] = design
    c7._GLOBAL["observed"] = observed
    c7._GLOBAL["R_z"] = R_z
    c7._GLOBAL["L_z"] = L_z
    c7._GLOBAL["n_common"] = len(common_idx)
    c7._GLOBAL["common_pos"] = common_pos
    c7._GLOBAL["max_len"] = max(observed[a]["resid_unr"].shape[0] for a in ASSETS)
    c7.install_simulation_calendar(design)
    nu_unr, nu_null = c9._install_nu(observed)
    recursive_audit = c9._fixed_dgp_audit_payload(SPEC)
    recursive_provenance = _recursive_audit_payload(recursive_audit)
    print(f"  fitted nu (null) = {np.round(nu_null, 3)}")
    print(
        f"  union calendar n={c7._GLOBAL['n_calendar']}; fixed/recursive "
        "analysis fingerprints="
        f"{str(recursive_audit['audit_analysis_stack_sha256'])[:16]}.../"
        f"{str(recursive_provenance['recursive_analysis_stack_sha256'])[:16]}..."
    )

    if TELEMETRY_ONLY:
        _replay_and_commit_floor_telemetry(d_bar_obs)
        print(f"\nTotal {time.time()-t_start:.1f}s")
        print("C19_RECURSIVE_TELEMETRY_DONE")
        return

    # implied unconditional variance under the null fit vs sample variance.
    # NB (round-6 audit fix): the EXPECTED GJR recursion under symmetric
    # innovations uses the SIGNED gamma/2 (E[eps^2 1(eps<0)] = sigma^2/2), so
    # variance-persistence = alpha + beta + gamma/2. The absolute-value form
    # alpha + beta + |gamma|/2 is the estimator's CONSTRAINT expression, not
    # the expectation; with gamma < 0 it overstates persistence. Both printed.
    print("\nPer-asset: implied unconditional var (recursion, signed gamma/2) vs sample var:")
    for a in ASSETS:
        pn = observed[a]["params_null"]
        persist_con = pn[1] + pn[3] + abs(pn[2]) / 2       # constraint form
        persist_var = pn[1] + pn[3] + pn[2] / 2            # expectation form
        base = pn[0] + float(np.mean(design[a]["exog_null"] @ pn[5:]))
        uncond = base / max(1e-6, (1 - persist_var))
        samp = float(np.var(design[a]["returns"]))
        print(f"  {a}: persist_constraint={persist_con:.4f}  persist_variance={persist_var:.4f}"
              f"  implied_uncond={uncond:9.1f}  sample_var={samp:6.1f}  ratio={uncond/samp:5.1f}x")

    print(f"\n[RECURSIVE t-copula NULL-imposed] B={B} n_jobs={N_JOBS}...")
    t0 = time.time()
    draw_seeds = np.arange(
        BASE_SEED + DRAW_SEED_OFFSET,
        BASE_SEED + DRAW_SEED_OFFSET + B,
        dtype=np.int64,
    )
    with Pool(processes=N_JOBS) as pool:
        results = pool.map(
            _draw_recursive_null_t,
            draw_seeds.tolist(),
            chunksize=max(1, B // (N_JOBS * 4)),
        )
    results = np.array(results, dtype=float)   # (B, 23); see _draw_recursive_null_t
    if results.shape != (B, 23):
        raise RuntimeError(f"unexpected recursive result shape {results.shape}")
    dbar = results[:, 0]
    exploded = results[:, 1] > 0.5
    max_sig2 = results[:, 2]
    mean_rv = results[:, 3]
    rescue_attempted = results[:, 4] > 0.5
    rescue_retained = results[:, 5] > 0.5
    asset_rescue_attempts = results[:, 6].astype(int)
    asset_rescue_successes = results[:, 7].astype(int)
    floor_hits = results[:, 8].astype(np.int64)
    floor_asset_paths = results[:, 9].astype(np.int64)
    min_raw_variance = results[:, 10]
    floor_hits_by_asset = results[:, 11:17].astype(np.int64)
    min_raw_variance_by_asset = results[:, 17:23]
    usable = ~np.isnan(dbar)
    if np.any(asset_rescue_attempts < 0) or np.any(asset_rescue_successes < 0):
        raise RuntimeError("recursive rescue telemetry contains negative counts")
    if np.any(asset_rescue_successes > asset_rescue_attempts):
        raise RuntimeError("recursive rescued-asset counts exceed attempts")
    if not np.array_equal(rescue_attempted, asset_rescue_attempts > 0):
        raise RuntimeError("recursive draw rescue flags disagree with asset attempts")
    if not np.array_equal(rescue_retained, rescue_attempted & usable):
        raise RuntimeError("recursive rescue-retention flags disagree with usable draws")
    if np.any(exploded & usable):
        raise RuntimeError("recursion-explosion draws must be excluded before refitting")
    if np.any(floor_hits < 0) or np.any(floor_hits_by_asset < 0):
        raise RuntimeError("variance-floor telemetry contains negative counts")
    if not np.array_equal(floor_hits, floor_hits_by_asset.sum(axis=1)):
        raise RuntimeError("draw-level and per-asset variance-floor counts disagree")
    if not np.array_equal(
        floor_asset_paths, (floor_hits_by_asset > 0).sum(axis=1)
    ):
        raise RuntimeError("variance-floor asset-path counts disagree")
    refit_excluded = (~usable) & (~exploded)
    n_total_exclusions = int((~usable).sum())
    n_recursion_explosion_exclusions = int(exploded.sum())
    n_refit_exclusions_after_rescue = int(refit_excluded.sum())
    n_draws_with_rescue_attempted = int(rescue_attempted.sum())
    n_draws_retained_after_rescue = int(rescue_retained.sum())
    n_draws_failed_after_rescue = int((rescue_attempted & refit_excluded).sum())
    n_asset_refits_with_rescue_attempted = int(asset_rescue_attempts.sum())
    n_asset_refits_rescued = int(asset_rescue_successes.sum())
    null_rec = dbar[usable]
    Bu = len(null_rec)
    print(f"  done {time.time()-t0:.1f}s  used={Bu}  total exclusions={n_total_exclusions}"
          f" ({n_total_exclusions/B:.2%})")
    print(f"  recursion-explosion exclusions={n_recursion_explosion_exclusions};"
          f" refit exclusions after shared rescue={n_refit_exclusions_after_rescue}")
    print(f"  draws with six-start rescue attempted={n_draws_with_rescue_attempted};"
          f" retained after rescue={n_draws_retained_after_rescue};"
          f" failed after rescue={n_draws_failed_after_rescue}")
    print(f"  asset refits with rescue attempted={n_asset_refits_with_rescue_attempted};"
          f" asset refits rescued={n_asset_refits_rescued}")
    floor_summary = _variance_floor_summary(
        dbar, d_bar_obs, floor_hits, floor_asset_paths, min_raw_variance,
        floor_hits_by_asset, min_raw_variance_by_asset,
    )
    print(
        "  variance-floor telemetry: "
        f"draws={floor_summary['n_draws_with_variance_floor_hit']}/{B}; "
        f"asset-days={floor_summary['n_variance_floor_hits']}; "
        f"asset paths={floor_summary['n_asset_paths_with_variance_floor_hit']}; "
        f"min raw variance={floor_summary['min_raw_variance_before_floor']:.6g}"
    )

    hits = int(np.sum(null_rec >= d_bar_obs))
    p_one = (hits + 1) / (Bu + 1)
    p_abs = (int(np.sum(np.abs(null_rec) >= abs(d_bar_obs))) + 1) / (Bu + 1)
    lower = (int(np.sum(null_rec <= d_bar_obs)) + 1) / (Bu + 1)
    p_eq = min(1.0, 2 * min(p_one, lower))
    p_worst_low = (hits + 1) / (B + 1)
    p_worst_high = (hits + n_total_exclusions + 1) / (B + 1)

    print(f"  RECURSIVE one-sided p={p_one:.4f}  abs-stat p={p_abs:.4f}  equal-tail p={p_eq:.4f}")
    print(f"  worst-case envelope: [{p_worst_low:.4f}, {p_worst_high:.4f}]")
    print(f"  null d* mean={null_rec.mean():.4f}  SD={null_rec.std():.4f}")

    # fixed-scheme reference distribution (paired seed schema) from c9's npz
    fixed_path = c2.OUT_DIR / f"c9-tcopula-draws-{SPEC}.npz"
    if not fixed_path.exists():
        raise FileNotFoundError(
            f"{fixed_path} is required; run c9 under seed schema v2 before c19"
        )
    with np.load(fixed_path) as archive:
        required = {
            "null_t", "null_t_full", "null_t_usable_mask",
            "null_t_excluded_mask", "draw_seeds", "B_requested",
            "base_seed", "draw_seed_offset", "first_draw_seed",
            "last_draw_seed", "seed_schema_version", "d_bar_obs",
            "draw_had_six_start_rescue_attempt_t",
            "draw_retained_after_six_start_rescue_t",
            "n_asset_refits_with_six_start_rescue_attempt_t",
            "n_asset_refits_rescued_by_six_start_t",
            "audit_design_sha256", "audit_fixed_null_dgp_sha256",
            "audit_refitter_sha256", "audit_simulator_sha256",
            "audit_analysis_stack_sha256",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"c9 fixed archive lacks fields: {sorted(missing)}")
        fixed_payload = {key: archive[key] for key in required}
    if int(fixed_payload["seed_schema_version"]) != 2:
        raise RuntimeError("c9 fixed archive is not union-calendar seed schema v2")
    if int(fixed_payload["B_requested"]) != B:
        raise RuntimeError("c9/c19 requested draw counts differ")
    expected_fixed_seeds = np.arange(
        BASE_SEED + DRAW_SEED_OFFSET,
        BASE_SEED + DRAW_SEED_OFFSET + B,
        dtype=np.int64,
    )
    if (
        int(fixed_payload["base_seed"]) != BASE_SEED
        or int(fixed_payload["draw_seed_offset"]) != DRAW_SEED_OFFSET
        or not np.array_equal(fixed_payload["draw_seeds"], expected_fixed_seeds)
        or int(fixed_payload["first_draw_seed"]) != int(expected_fixed_seeds[0])
        or int(fixed_payload["last_draw_seed"]) != int(expected_fixed_seeds[-1])
    ):
        raise RuntimeError("c9/c19 exact draw-seed schemas differ")
    fixed_d_bar_obs = float(fixed_payload["d_bar_obs"])
    if fixed_d_bar_obs != d_bar_obs:
        raise RuntimeError(
            "c9/c19 observed fits are not exactly paired: "
            f"c9 d_bar_obs={fixed_d_bar_obs:.17g}, c19={d_bar_obs:.17g}"
        )
    for key in (
        "audit_design_sha256", "audit_fixed_null_dgp_sha256",
        "audit_refitter_sha256", "audit_simulator_sha256",
        "audit_analysis_stack_sha256",
    ):
        if str(fixed_payload[key]) != str(recursive_audit[key]):
            raise RuntimeError(f"c9/c19 {key} differs; paired comparison invalid")
    null_fix_full = np.asarray(fixed_payload["null_t_full"], dtype=float)
    fixed_usable = np.asarray(fixed_payload["null_t_usable_mask"], dtype=bool)
    fixed_excluded = np.asarray(fixed_payload["null_t_excluded_mask"], dtype=bool)
    null_fix = np.asarray(fixed_payload["null_t"], dtype=float)
    if (
        null_fix_full.shape != (B,)
        or not np.array_equal(fixed_excluded, ~fixed_usable)
        or not np.array_equal(fixed_usable, ~np.isnan(null_fix_full))
        or not np.array_equal(null_fix, null_fix_full[fixed_usable])
    ):
        raise RuntimeError("c9 fixed full arrays, masks, and compression disagree")
    fixed_rescue_attempted = np.asarray(
        fixed_payload["draw_had_six_start_rescue_attempt_t"], dtype=bool
    )
    fixed_rescue_retained = np.asarray(
        fixed_payload["draw_retained_after_six_start_rescue_t"], dtype=bool
    )
    fixed_asset_rescue_attempts = np.asarray(
        fixed_payload["n_asset_refits_with_six_start_rescue_attempt_t"], dtype=np.int64
    )
    fixed_asset_rescue_successes = np.asarray(
        fixed_payload["n_asset_refits_rescued_by_six_start_t"], dtype=np.int64
    )
    for name, values in {
        "rescue_attempted": fixed_rescue_attempted,
        "rescue_retained": fixed_rescue_retained,
        "asset_rescue_attempts": fixed_asset_rescue_attempts,
        "asset_rescue_successes": fixed_asset_rescue_successes,
    }.items():
        if values.shape != (B,):
            raise RuntimeError(f"c9 fixed {name} telemetry has wrong shape")
    if not np.array_equal(
        fixed_rescue_retained, fixed_rescue_attempted & fixed_usable
    ):
        raise RuntimeError("c9 fixed rescue-retention telemetry is inconsistent")
    if np.any(fixed_asset_rescue_attempts < 0) or np.any(
        fixed_asset_rescue_successes < 0
    ):
        raise RuntimeError("c9 fixed rescue telemetry contains negative counts")
    if np.any(fixed_asset_rescue_successes > fixed_asset_rescue_attempts):
        raise RuntimeError("c9 fixed rescued-asset counts exceed attempts")
    if not np.array_equal(
        fixed_rescue_attempted, fixed_asset_rescue_attempts > 0
    ):
        raise RuntimeError("c9 fixed rescue flags disagree with asset attempts")
    print(f"\n  FIXED (c9 npz): used={len(null_fix)}  mean={null_fix.mean():.4f}"
          f"  SD={null_fix.std():.4f}"
          f"  one-sided p={(np.sum(null_fix >= d_bar_obs)+1)/(len(null_fix)+1):.4f}")

    # failure-selection diagnostics
    print("\nFailure-selection diagnostics:")
    for label, mask in [("usable", usable),
                        ("refit-excluded", refit_excluded),
                        ("explosion-excluded", exploded)]:
        if mask.sum():
            print(f"  {label:18s}: n={int(mask.sum()):4d}  max_sig2 median={np.median(max_sig2[mask]):10.1f}"
                  f"  p90={np.percentile(max_sig2[mask], 90):12.1f}"
                  f"  realized-var median={np.median(mean_rv[mask]):8.1f}")
    samp_vars = [float(np.var(design[a]['returns'])) for a in ASSETS]
    print(f"  ACTUAL sample vars: {np.round(samp_vars, 1)} (mean {np.mean(samp_vars):.1f})")
    print(f"  recursive realized-var (usable draws): median {np.median(mean_rv[usable]):.1f},"
          f"  IQR [{np.percentile(mean_rv[usable], 25):.1f}, {np.percentile(mean_rv[usable], 75):.1f}]")

    if len(null_fix):
        fixed_upper_hits = int(np.sum(null_fix >= d_bar_obs))
        fixed_abs_hits = int(np.sum(np.abs(null_fix) >= abs(d_bar_obs)))
        fixed_lower_hits = int(np.sum(null_fix <= d_bar_obs))
        p_one_fixed = (fixed_upper_hits + 1) / (len(null_fix) + 1)
        p_abs_fixed = (fixed_abs_hits + 1) / (len(null_fix) + 1)
        p_equal_tail_fixed = min(
            1.0,
            2 * min(p_one_fixed, (fixed_lower_hits + 1) / (len(null_fix) + 1)),
        )
    else:
        fixed_upper_hits = fixed_abs_hits = fixed_lower_hits = 0
        p_one_fixed = p_abs_fixed = p_equal_tail_fixed = np.nan

    draws_path = c2.OUT_DIR / f"c19-recursive-draws-{SPEC}.npz"
    draws_payload = dict(
        dbar=dbar,
        draw_seeds=draw_seeds,
        recursive_usable_mask=usable,
        recursive_excluded_mask=~usable,
        B_requested=np.int64(B),
        # Keep the legacy key while also exposing the exclusion meaning clearly.
        exploded=exploded,
        recursion_explosion_excluded=exploded,
        draw_had_six_start_rescue_attempt=rescue_attempted,
        draw_retained_after_six_start_rescue=rescue_retained,
        n_asset_refits_with_six_start_rescue_attempt=asset_rescue_attempts,
        n_asset_refits_rescued_by_six_start=asset_rescue_successes,
        max_sig2=max_sig2,
        mean_rv=mean_rv,
        sample_vars_actual=np.asarray(samp_vars, dtype=float),
        sample_var_mean_actual=np.asarray(float(np.mean(samp_vars))),
        draw_had_variance_floor_hit=floor_hits > 0,
        n_variance_floor_hits=floor_hits,
        n_asset_paths_with_variance_floor_hit=floor_asset_paths,
        min_raw_variance_before_floor=min_raw_variance,
        variance_floor_hits_by_asset=floor_hits_by_asset,
        min_raw_variance_by_asset=min_raw_variance_by_asset,
        variance_floor_asset_names=np.asarray(ASSETS),
        d_bar_obs=d_bar_obs,
        null_fixed=null_fix,
        null_fixed_full=null_fix_full,
        null_fixed_usable_mask=fixed_usable,
        null_fixed_excluded_mask=fixed_excluded,
        fixed_draw_seeds=expected_fixed_seeds,
        fixed_seed_schema_version=np.int64(2),
        fixed_base_seed=np.int64(BASE_SEED),
        fixed_draw_seed_offset=np.int64(DRAW_SEED_OFFSET),
        fixed_first_draw_seed=np.int64(expected_fixed_seeds[0]),
        fixed_last_draw_seed=np.int64(expected_fixed_seeds[-1]),
        fixed_draw_had_six_start_rescue_attempt=fixed_rescue_attempted,
        fixed_draw_retained_after_six_start_rescue=fixed_rescue_retained,
        fixed_n_asset_refits_with_six_start_rescue_attempt=fixed_asset_rescue_attempts,
        fixed_n_asset_refits_rescued_by_six_start=fixed_asset_rescue_successes,
        fixed_audit_design_sha256=recursive_audit["audit_design_sha256"],
        fixed_audit_fixed_null_dgp_sha256=recursive_audit[
            "audit_fixed_null_dgp_sha256"
        ],
        fixed_audit_refitter_sha256=recursive_audit["audit_refitter_sha256"],
        fixed_audit_simulator_sha256=recursive_audit["audit_simulator_sha256"],
        fixed_audit_analysis_stack_sha256=recursive_audit[
            "audit_analysis_stack_sha256"
        ],
        fixed_audit_locally_reconstructed_equal=np.bool_(True),
        base_seed=np.int64(BASE_SEED),
        first_draw_seed=np.int64(draw_seeds[0]),
        last_draw_seed=np.int64(draw_seeds[-1]),
        **recursive_provenance,
        **recursive_audit,
    )
    _atomic_savez(draws_path, draws_payload)
    row = {
        "spec": SPEC,
        "base_seed": BASE_SEED,
        "recursive_seed_schema_version": RECURSIVE_SEED_SCHEMA_VERSION,
        "recursive_draw_seed_offset": DRAW_SEED_OFFSET,
        "first_draw_seed": int(draw_seeds[0]),
        "last_draw_seed": int(draw_seeds[-1]),
        "design_sha256": str(recursive_audit["audit_design_sha256"]),
        "fixed_null_dgp_sha256": str(
            recursive_audit["audit_fixed_null_dgp_sha256"]
        ),
        "refitter_sha256": str(recursive_audit["audit_refitter_sha256"]),
        "simulator_sha256": str(recursive_audit["audit_simulator_sha256"]),
        "analysis_stack_sha256": str(
            recursive_audit["audit_analysis_stack_sha256"]
        ),
        "recursive_simulator_sha256": str(
            recursive_provenance["recursive_simulator_sha256"]
        ),
        "recursive_analysis_stack_sha256": str(
            recursive_provenance["recursive_analysis_stack_sha256"]
        ),
        "n_calendar": int(c7._GLOBAL["n_calendar"]),
        "availability_pattern_codes": ";".join(
            str(int(v)) for v in c7._GLOBAL["availability_pattern_codes"]
        ),
        "availability_pattern_counts": ";".join(
            str(int(v)) for v in c7._GLOBAL["availability_pattern_counts"]
        ),
        "B": B,
        "B_requested": B,
        "B_used": Bu,
        "n_total_exclusions": n_total_exclusions,
        "n_recursion_explosion_exclusions": n_recursion_explosion_exclusions,
        "n_refit_exclusions_after_six_start_rescue": n_refit_exclusions_after_rescue,
        "n_draws_with_six_start_rescue_attempt": n_draws_with_rescue_attempted,
        "n_draws_retained_after_six_start_rescue": n_draws_retained_after_rescue,
        "n_draws_failed_after_six_start_rescue": n_draws_failed_after_rescue,
        "n_asset_refits_with_six_start_rescue_attempt": n_asset_refits_with_rescue_attempted,
        "n_asset_refits_rescued_by_six_start": n_asset_refits_rescued,
        "fraction_total_excluded": n_total_exclusions / B,
        "fraction_recursion_explosion_excluded": n_recursion_explosion_exclusions / B,
        "fraction_refit_excluded_after_six_start_rescue": n_refit_exclusions_after_rescue / B,
        "d_bar_obs": d_bar_obs,
        "n_upper_tail_hits_recursive": hits,
        "n_abs_stat_hits_recursive": int(np.sum(np.abs(null_rec) >= abs(d_bar_obs))),
        "n_lower_tail_hits_recursive": int(np.sum(null_rec <= d_bar_obs)),
        "p_one_recursive": p_one, "p_abs_recursive": p_abs, "p_equal_tail_recursive": p_eq,
        "p_worst_low": p_worst_low, "p_worst_high": p_worst_high,
        "null_mean_recursive": float(null_rec.mean()), "null_sd_recursive": float(null_rec.std()),
        "B_used_fixed": len(null_fix),
        "B_requested_fixed": B,
        "n_total_exclusions_fixed": int(fixed_excluded.sum()),
        "fraction_total_excluded_fixed": float(fixed_excluded.mean()),
        "n_draws_with_six_start_rescue_attempt_fixed": int(
            fixed_rescue_attempted.sum()
        ),
        "n_draws_retained_after_six_start_rescue_fixed": int(
            fixed_rescue_retained.sum()
        ),
        "n_draws_failed_after_six_start_rescue_fixed": int(
            (fixed_rescue_attempted & fixed_excluded).sum()
        ),
        "n_asset_refits_with_six_start_rescue_attempt_fixed": int(
            fixed_asset_rescue_attempts.sum()
        ),
        "n_asset_refits_rescued_by_six_start_fixed": int(
            fixed_asset_rescue_successes.sum()
        ),
        "design_sha256_fixed": str(recursive_audit["audit_design_sha256"]),
        "fixed_null_dgp_sha256_fixed": str(
            recursive_audit["audit_fixed_null_dgp_sha256"]
        ),
        "refitter_sha256_fixed": str(recursive_audit["audit_refitter_sha256"]),
        "simulator_sha256_fixed": str(recursive_audit["audit_simulator_sha256"]),
        "analysis_stack_sha256_fixed": str(
            recursive_audit["audit_analysis_stack_sha256"]
        ),
        "fixed_audit_locally_reconstructed_equal": True,
        "n_upper_tail_hits_fixed": fixed_upper_hits,
        "n_abs_stat_hits_fixed": fixed_abs_hits,
        "n_lower_tail_hits_fixed": fixed_lower_hits,
        "p_one_fixed": p_one_fixed,
        "p_abs_fixed": p_abs_fixed,
        "p_equal_tail_fixed": p_equal_tail_fixed,
        "null_mean_fixed": float(null_fix.mean()) if len(null_fix) else np.nan,
        "null_sd_fixed": float(null_fix.std()) if len(null_fix) else np.nan,
        "realized_var_median_recursive": float(np.median(mean_rv[usable])),
        "sample_var_mean_actual": float(np.mean(samp_vars)),
        "runtime_s": time.time() - t_start,
    }
    row.update(floor_summary)
    csv_path = c2.OUT_DIR / f"c19-recursive-sensitivity-{SPEC}.csv"
    csv_tmp = csv_path.with_name(csv_path.stem + ".write-tmp.csv")
    pd.DataFrame([row]).to_csv(csv_tmp, index=False)
    csv_trial = pd.read_csv(csv_tmp)
    if len(csv_trial) != 1 or list(csv_trial.columns) != list(row):
        raise RuntimeError(f"temporary {csv_path.name} failed schema validation")
    if str(csv_trial.iloc[0]["spec"]) != SPEC:
        raise RuntimeError(f"temporary {csv_path.name} lost the specification label")
    csv_tmp.replace(csv_path)
    if SPEC == "baseline":
        draws_alias = c2.OUT_DIR / "c19-recursive-draws.npz"
        csv_alias = c2.OUT_DIR / "c19-recursive-sensitivity.csv"
        shutil.copyfile(draws_path, draws_alias)
        shutil.copyfile(csv_path, csv_alias)
        print(f"  refreshed baseline aliases: {draws_alias.name}, {csv_alias.name}")
    if _write_finding_if_complete():
        print("  refreshed c19 two-spec finding")
    else:
        print("  c19 two-spec finding deferred until both current-schema runs exist")
    print(f"\nTotal {time.time()-t_start:.1f}s")
    print("C19_RECURSIVE_DONE")


if __name__ == "__main__":
    _args = _parse_cli()
    SPEC = _args.spec
    TELEMETRY_ONLY = _args.telemetry_only
    SYNC_FIXED_ONLY = _args.sync_fixed_only
    main()
