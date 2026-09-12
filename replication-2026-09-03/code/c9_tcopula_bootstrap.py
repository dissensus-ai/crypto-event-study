"""
C9: Student-t-copula CCC-GARCH-X fixed-path bootstrap.
======================================================

Why this exists
---------------
c7 / c8h were intermediate cross-asset bootstrap tests for the
infrastructure-vs-regulatory variance-coefficient asymmetry. A pre-submission
review caught a misspecification in both: the CCC parametric bootstrap draws the
standardised innovations as GAUSSIAN

    Z = rng.standard_normal((n, 6)) @ L.T          # c7 lines ~204/213/229/234

even though every per-asset GJR-GARCH-X is FITTED with Student-t errors, nu ~
3.1-4.6 (heavy tails). An unstandardised t_nu variate has variance nu/(nu-2),
but the fitted model uses z = sqrt((nu-2)/nu) * u, so both fitted z and the
Gaussian comparator have unit variance. The misspecification is therefore the
distributional shape (more body mass plus rare extremes) and the absence of
joint tail dependence, not a variance-scale mismatch. In the fitted internal-
calibration experiment this Gaussian rule is anti-conservative. The direction
of the p-value change is an empirical result of this design, not a universal
property of replacing Gaussian innovations.

The fix
-------
Replace the Gaussian innovation draw with a STUDENT-t COPULA:

  1. Z ~ MVN(0, R_z) via the existing Cholesky L of the standardised-residual
     correlation R_z (cross-asset dependence, unchanged).
  2. (true t-copula) divide by a SHARED chi-square mixing variable:
         W ~ chi2(nu_c) / nu_c,   T = Z / sqrt(W)
     so the latent vector T has a multivariate-t(nu_c, R_z) distribution -- this
     restores JOINT TAIL DEPENDENCE (the Gaussian copula has none), addressing
     the related zero-tail-dependence concern. nu_c = median fitted nu.
  3. Map to uniforms with the t_{nu_c} CDF:  U = F_{t,nu_c}(T)  (in [0,1]).
  4. Map each asset's column to a Student-t marginal with THAT ASSET'S FITTED nu
     via the t quantile function:  innov_j = t_{nu_j}.ppf(U_j).
  5. Rescale to UNIT VARIANCE:  innov_j *= sqrt((nu_j - 2)/nu_j), so the
     innovation feeds eps = innov * sqrt(sigma2_t) at unit scale (the variance
     recursion expects unit-variance innovations -- sigma2_t already carries the
     scale).

Margins are exact per-asset Student-t with the fitted nu; the copula carries
both the fitted cross-asset correlation AND tail dependence. One full six-vector
is drawn on every date in the union asset calendar; unavailable coordinates are
discarded, so pre-BNB dates use the corresponding five-dimensional marginal of
the six-way-overlap R_z rather than independent fallback draws. Applied
identically in the NULL-imposed draw and the unrestricted (CI) draw, and to the
break-controlled c8h specs.

Everything else follows the c7/c8h fixed-path design: B=2000, null imposes
delta_infra=delta_reg via the combined dummy, D_infra/D_reg kept first as exog
[5]/[6], same convergence/degeneracy guards (DELTA_CAP=50), same drop-rate
reporting. We import c7's refit and union-calendar engines so the simulation
calendar, seed mapping, screening, and aggregation paths are shared explicitly.

Run:
    python c9_tcopula_bootstrap.py --B 2000 --n_jobs 22

Outputs (the package `results/` directory unless `CES_OUT_DIR` is set):
    c9-tcopula-results.csv
    c9-tcopula-bootstrap-FINDING.md   (saved by the team-lead-requested name)
    c9-tcopula-draws-{baseline,crisis,full}.npz (full seed-indexed draws,
    masks, calendar/design fingerprints, and legacy compressed arrays)
"""
import argparse
import hashlib
import inspect
import time
import warnings
import multiprocessing as mp
from pathlib import Path

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
import c8h_break_controls_ccc_bootstrap as c8h
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA, _variance_recursion_core

ASSETS = c2.ASSETS

# Set False to use a Gaussian copula with Student-t margins (the literal spec the
# Gaussian-copula mapping U = Phi(Z)); True uses a multivariate-t copula
# (shared chi-square mixing -> joint tail dependence). We report the latter as
# the fitted heavy-tail specification and the Gaussian copula as a comparator.
USE_T_COPULA = True


# ---------------------------------------------------------------------------
# Student-t innovation builder shared by both draw functions.
# ---------------------------------------------------------------------------
def _student_t_innovations(rng, L, nu_vec, nu_c, size_n):
    """
    Draw an (size_n x n_assets) array of innovations with:
      - cross-asset dependence given by L (Cholesky of R_z),
      - per-asset Student-t marginals with df nu_vec[j],
      - UNIT VARIANCE per column,
      - (if USE_T_COPULA) joint tail dependence via a true t-copula.

    Returns the innovation array; the caller scatters columns onto each asset's
    full-length series (common positions) exactly as c7 did with the Gaussian
    draw.
    """
    k = len(nu_vec)
    Z = rng.standard_normal((size_n, k)) @ L.T          # MVN(0, R_z), corr ~ R_z
    if USE_T_COPULA:
        # shared chi-square mixing variable -> latent ~ multivariate-t(nu_c, R_z)
        w = rng.chisquare(nu_c, size=(size_n, 1)) / nu_c
        T = Z / np.sqrt(w)
        U = stats.t.cdf(T, nu_c)                         # uniform margins, t-copula
    else:
        U = stats.norm.cdf(Z)                            # uniform margins, gaussian copula
    U = np.clip(U, 1e-12, 1.0 - 1e-12)                   # guard t.ppf at the tails
    innov = np.empty_like(U)
    for j in range(k):
        nu_j = nu_vec[j]
        t_j = stats.t.ppf(U[:, j], nu_j)                 # Student-t marginal, fitted nu
        innov[:, j] = t_j * np.sqrt((nu_j - 2.0) / nu_j)  # rescale to unit variance
    return innov


def _student_t_calendar_innovations(rng, nu_vec, nu_c, state=None):
    """Draw one joint six-vector per union-calendar date, then scatter it.

    Unavailable coordinates are discarded.  Thus a date with five observed
    assets has the exact five-dimensional marginal implied by the corresponding
    submatrix of the six-way-overlap ``R_z``.  The shared chi-square mixer is
    drawn once per calendar date and shared across all six latent coordinates.
    """
    source = c7._GLOBAL if state is None else state
    innovations = _student_t_innovations(
        rng,
        np.asarray(source["L_z"], dtype=float),
        np.asarray(nu_vec, dtype=float),
        float(nu_c),
        int(source["n_calendar"]),
    )
    return c7.scatter_calendar_innovations(innovations, state=source)


# ---------------------------------------------------------------------------
# Patched draw functions. Identical to c7's except the innovation source. They
# read nu from c7._GLOBAL["nu_null"] / ["nu_unr"] (per-asset fitted df) and the
# shared copula df from c7._GLOBAL["nu_c_null"] / ["nu_c_unr"].
# ---------------------------------------------------------------------------
def _draw_parametric_null_t_core(seed, return_diagnostics=False):
    """Simulate and refit one fixed-path null draw.

    ``_draw_parametric_null_t`` retains the historical scalar interface used by
    c20 and other callers.  The diagnostic wrapper opts into c7's refit
    telemetry without changing the simulated panel, seed, optimiser starts, or
    returned statistic.
    """
    rng = np.random.default_rng(seed)
    design = c7._GLOBAL["design"]
    obs = c7._GLOBAL["observed"]
    nu_vec = c7._GLOBAL["nu_null"]          # per-asset null-fit nu
    nu_c = c7._GLOBAL["nu_c_null"]

    z_by_asset = _student_t_calendar_innovations(rng, nu_vec, nu_c)
    returns_by_asset = {}
    for a in ASSETS:
        sig2 = obs[a]["sigma2_null"]
        mean_r = obs[a]["mean_return"]
        eps = z_by_asset[a] * np.sqrt(sig2)
        returns_by_asset[a] = mean_r + eps
    return c7._refit_unrestricted_dbar(
        returns_by_asset, return_diagnostics=return_diagnostics
    )


def _draw_parametric_null_t(args):
    """Historical scalar draw interface (kept for c20 and external callers)."""
    return _draw_parametric_null_t_core(args, return_diagnostics=False)


def _draw_parametric_null_t_diagnostics(args):
    """Diagnostic draw interface used only by c9's full-length runner."""
    return _draw_parametric_null_t_core(args, return_diagnostics=True)


def _draw_parametric_unr_t(args):
    seed = args
    rng = np.random.default_rng(seed)
    obs = c7._GLOBAL["observed"]
    nu_vec = c7._GLOBAL["nu_unr"]           # per-asset unrestricted-fit nu
    nu_c = c7._GLOBAL["nu_c_unr"]

    z_by_asset = _student_t_calendar_innovations(rng, nu_vec, nu_c)
    returns_by_asset = {}
    for a in ASSETS:
        sig2 = obs[a]["sigma2_unr"]
        mean_r = obs[a]["mean_return"]
        eps = z_by_asset[a] * np.sqrt(sig2)
        returns_by_asset[a] = mean_r + eps
    return c7._refit_unrestricted_dbar(returns_by_asset)


# ---------------------------------------------------------------------------
# Populate the per-asset / shared nu into c7._GLOBAL given an `observed` dict.
# ---------------------------------------------------------------------------
def _install_nu(observed):
    nu_unr = np.array([observed[a]["params_unr"][4] for a in ASSETS])
    nu_null = np.array([observed[a]["params_null"][4] for a in ASSETS])
    c7._GLOBAL["nu_unr"] = nu_unr
    c7._GLOBAL["nu_null"] = nu_null
    # shared copula df for the true t-copula: median fitted nu (robust, heavy-tail)
    c7._GLOBAL["nu_c_unr"] = float(np.median(nu_unr))
    c7._GLOBAL["nu_c_null"] = float(np.median(nu_null))
    return nu_unr, nu_null


def _concat_ragged(arrays):
    """Concatenate first-axis-ragged arrays and return exact row offsets.

    Most specifications give every asset the same number of exogenous columns.
    In the full-regime specification BNB has no pre-listing regime column, so
    its design matrix is one column narrower.  Pad only that heterogeneous
    trailing dimension with NaN; equal-width inputs follow the original path
    exactly, preserving the existing baseline/crisis audit bytes and hashes.
    """
    arrays = [np.asarray(array) for array in arrays]
    lengths = np.asarray([len(array) for array in arrays], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    if arrays:
        trailing_shapes = {array.shape[1:] for array in arrays}
        if len(trailing_shapes) > 1:
            if any(array.ndim != 2 for array in arrays):
                raise ValueError(
                    "heterogeneous trailing shapes are supported only for matrices"
                )
            width = max(array.shape[1] for array in arrays)
            padded = []
            for array in arrays:
                block = np.full((len(array), width), np.nan, dtype=np.float64)
                block[:, :array.shape[1]] = array
                padded.append(block)
            arrays = padded
        concatenated = np.concatenate(arrays, axis=0)
    else:
        concatenated = np.asarray([], dtype=float)
    return concatenated, offsets


def _stack_ragged_vectors(arrays):
    """Stack per-asset vectors, padding only genuinely shorter vectors."""
    arrays = [np.asarray(array, dtype=np.float64) for array in arrays]
    if not arrays:
        return np.empty((0, 0), dtype=np.float64)
    lengths = {len(array) for array in arrays}
    if len(lengths) == 1:
        return np.stack(arrays)
    width = max(lengths)
    stacked = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for row, array in enumerate(arrays):
        stacked[row, :len(array)] = array
    return stacked


def _hash_named_arrays(named_arrays):
    """Platform-stable SHA-256 over labels, dtypes, shapes, and C-order bytes."""
    digest = hashlib.sha256()
    for label, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        label_bytes = str(label).encode("utf-8")
        dtype_bytes = array.dtype.str.encode("ascii")
        digest.update(np.asarray([len(label_bytes)], dtype="<i8").tobytes())
        digest.update(label_bytes)
        digest.update(np.asarray([len(dtype_bytes)], dtype="<i8").tobytes())
        digest.update(dtype_bytes)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fixed_dgp_audit_payload(spec):
    """Return reconstructible fixed-null inputs and deterministic fingerprints.

    Call only after c7's globals and the fitted degrees of freedom are installed.
    The payload archives the null parameter/path inputs, complete design arrays,
    dependence mapping, and the exact refit implementation contract.  c13 and
    c19 use this helper to assert that they are operating on the same fitted DGP
    and refitter rather than merely obtaining a similar observed statistic.
    """
    design = c7._GLOBAL["design"]
    observed = c7._GLOBAL["observed"]
    assets = np.asarray(ASSETS)

    returns_concat, row_offsets = _concat_ragged(
        [np.asarray(design[a]["returns"], dtype=np.float64) for a in ASSETS]
    )
    index_concat, index_offsets = _concat_ragged(
        [pd.DatetimeIndex(design[a]["index"]).asi8.astype(np.int64) for a in ASSETS]
    )
    exog_unr_concat, exog_unr_offsets = _concat_ragged(
        [np.asarray(design[a]["exog_unr"], dtype=np.float64) for a in ASSETS]
    )
    exog_null_concat, exog_null_offsets = _concat_ragged(
        [np.asarray(design[a]["exog_null"], dtype=np.float64) for a in ASSETS]
    )
    sigma2_null_concat, sigma2_null_offsets = _concat_ragged(
        [np.asarray(observed[a]["sigma2_null"], dtype=np.float64) for a in ASSETS]
    )
    common_pos_concat, common_pos_offsets = _concat_ragged(
        [np.asarray(c7._GLOBAL["common_pos"][a], dtype=np.int64) for a in ASSETS]
    )
    calendar_pos_concat, calendar_pos_offsets = _concat_ragged(
        [np.asarray(c7._GLOBAL["calendar_pos"][a], dtype=np.int64) for a in ASSETS]
    )
    simulation_calendar_ns = pd.DatetimeIndex(
        c7._GLOBAL["simulation_calendar"]
    ).asi8.astype(np.int64)
    availability_matrix = np.asarray(
        c7._GLOBAL["availability_matrix"], dtype=np.bool_
    )
    null_params = _stack_ragged_vectors(
        [np.asarray(observed[a]["params_null"], dtype=np.float64) for a in ASSETS]
    )
    mean_returns = np.asarray(
        [observed[a]["mean_return"] for a in ASSETS], dtype=np.float64
    )
    nu_null = np.asarray(c7._GLOBAL["nu_null"], dtype=np.float64)
    R_z = np.asarray(c7._GLOBAL["R_z"], dtype=np.float64)
    L_z = np.asarray(c7._GLOBAL["L_z"], dtype=np.float64)

    design_named = [
        ("asset_names", assets),
        ("returns_concat", returns_concat),
        ("row_offsets", row_offsets),
        ("index_ns_concat", index_concat),
        ("index_offsets", index_offsets),
        ("exog_unr_concat", exog_unr_concat),
        ("exog_unr_offsets", exog_unr_offsets),
        ("exog_null_concat", exog_null_concat),
        ("exog_null_offsets", exog_null_offsets),
        ("simulation_calendar_ns", simulation_calendar_ns),
        ("calendar_pos_concat", calendar_pos_concat),
        ("calendar_pos_offsets", calendar_pos_offsets),
        ("availability_matrix", availability_matrix),
    ]
    dgp_named = [
        ("asset_names", assets),
        ("null_params", null_params),
        ("mean_returns", mean_returns),
        ("sigma2_null_concat", sigma2_null_concat),
        ("sigma2_null_offsets", sigma2_null_offsets),
        ("common_pos_concat", common_pos_concat),
        ("common_pos_offsets", common_pos_offsets),
        ("simulation_calendar_ns", simulation_calendar_ns),
        ("calendar_pos_concat", calendar_pos_concat),
        ("calendar_pos_offsets", calendar_pos_offsets),
        ("nu_null", nu_null),
        ("nu_c_null", np.asarray(c7._GLOBAL["nu_c_null"], dtype=np.float64)),
        ("R_z", R_z),
        ("L_z", L_z),
        ("use_t_copula", np.asarray(USE_T_COPULA, dtype=np.bool_)),
    ]
    design_sha256 = _hash_named_arrays(design_named)
    fixed_null_dgp_sha256 = _hash_named_arrays(dgp_named)

    refitter_contract = (
        f"assets={','.join(ASSETS)};MAX_ITER={c7.MAX_ITER};"
        f"DELTA_CAP={c7.DELTA_CAP:.17g};rescue_starts=6;"
        f"rescue_seed={c7._GLOBAL.get('rescue_seed', 20260807)};"
        f"numba_enabled={_HAVE_NUMBA}\n"
        + inspect.getsource(c7._plausible_delta)
        + inspect.getsource(c7._refit_unrestricted_dbar)
        + inspect.getsource(_variance_recursion_core)
        + inspect.getsource(FastTARCHX)
    )
    refitter_sha256 = hashlib.sha256(refitter_contract.encode("utf-8")).hexdigest()
    simulator_contract = (
        "seed_schema_version=2;one_union_calendar_by_six_latent_draw_per_seed\n"
        + inspect.getsource(c7.install_simulation_calendar)
        + inspect.getsource(c7.scatter_calendar_innovations)
        + inspect.getsource(c7.draw_gaussian_calendar_innovations)
        + inspect.getsource(c7._draw_parametric_null)
        + inspect.getsource(_student_t_innovations)
        + inspect.getsource(_student_t_calendar_innovations)
        + inspect.getsource(_draw_parametric_null_t_core)
    )
    simulator_sha256 = hashlib.sha256(
        simulator_contract.encode("utf-8")
    ).hexdigest()
    analysis_stack_sha256 = hashlib.sha256(
        (spec + "\n" + design_sha256 + "\n" + fixed_null_dgp_sha256 + "\n"
         + refitter_sha256 + "\n" + simulator_sha256).encode("utf-8")
    ).hexdigest()

    return {
        "audit_spec": np.asarray(spec),
        "audit_asset_names": assets,
        "audit_design_returns_concat": returns_concat,
        "audit_design_row_offsets": row_offsets,
        "audit_design_index_ns_concat": index_concat,
        "audit_design_index_offsets": index_offsets,
        "audit_design_exog_unr_concat": exog_unr_concat,
        "audit_design_exog_unr_offsets": exog_unr_offsets,
        "audit_design_exog_null_concat": exog_null_concat,
        "audit_design_exog_null_offsets": exog_null_offsets,
        "audit_null_params_by_asset": null_params,
        "audit_null_mean_returns": mean_returns,
        "audit_null_sigma2_concat": sigma2_null_concat,
        "audit_null_sigma2_offsets": sigma2_null_offsets,
        "audit_common_pos_concat": common_pos_concat,
        "audit_common_pos_offsets": common_pos_offsets,
        "audit_simulation_calendar_ns": simulation_calendar_ns,
        "audit_calendar_pos_concat": calendar_pos_concat,
        "audit_calendar_pos_offsets": calendar_pos_offsets,
        "audit_availability_matrix": availability_matrix,
        "audit_availability_pattern_code": np.asarray(
            c7._GLOBAL["availability_pattern_code"], dtype=np.int64
        ),
        "audit_availability_pattern_codes": np.asarray(
            c7._GLOBAL["availability_pattern_codes"], dtype=np.int64
        ),
        "audit_availability_pattern_counts": np.asarray(
            c7._GLOBAL["availability_pattern_counts"], dtype=np.int64
        ),
        "audit_availability_pattern_first_date_ns": np.asarray(
            c7._GLOBAL["availability_pattern_first_date_ns"], dtype=np.int64
        ),
        "audit_availability_pattern_last_date_ns": np.asarray(
            c7._GLOBAL["availability_pattern_last_date_ns"], dtype=np.int64
        ),
        "audit_nu_null_by_asset": nu_null,
        "audit_nu_c_null": np.asarray(c7._GLOBAL["nu_c_null"], dtype=np.float64),
        "audit_R_z": R_z,
        "audit_L_z": L_z,
        "audit_use_t_copula": np.asarray(USE_T_COPULA, dtype=np.bool_),
        "audit_design_sha256": np.asarray(design_sha256),
        "audit_fixed_null_dgp_sha256": np.asarray(fixed_null_dgp_sha256),
        "audit_refitter_sha256": np.asarray(refitter_sha256),
        "audit_simulator_sha256": np.asarray(simulator_sha256),
        "audit_analysis_stack_sha256": np.asarray(analysis_stack_sha256),
        "audit_refitter_contract": np.asarray(refitter_contract),
        "audit_simulator_contract": np.asarray(simulator_contract),
    }


def _run_scalar_bootstrap_full(draw_fn, B, n_jobs, base_seed):
    """Run scalar draws while retaining every seed position, including NaNs."""
    seeds = np.arange(base_seed, base_seed + B, dtype=np.int64)
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(
            draw_fn, seeds.tolist(), chunksize=max(1, B // (n_jobs * 4))
        )
    full = np.asarray(results, dtype=float)
    if full.shape != (B,):
        raise RuntimeError(f"unexpected bootstrap result shape {full.shape}; expected {(B,)}")
    return seeds, full


def _run_t_bootstrap_full(B, n_jobs, base_seed):
    """Run t-copula draws with full seed order and one-rescue telemetry."""
    seeds = np.arange(base_seed, base_seed + B, dtype=np.int64)
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(
            _draw_parametric_null_t_diagnostics,
            seeds.tolist(),
            chunksize=max(1, B // (n_jobs * 4)),
        )
    if len(results) != B:
        raise RuntimeError(f"received {len(results)} bootstrap results; expected {B}")

    full = np.asarray([result[0] for result in results], dtype=float)
    diagnostics = [result[1] for result in results]
    telemetry = {
        "draw_had_six_start_rescue_attempt_t": np.asarray(
            [d["draw_had_six_start_rescue_attempt"] for d in diagnostics],
            dtype=bool,
        ),
        "draw_retained_after_six_start_rescue_t": np.asarray(
            [d["draw_retained_after_six_start_rescue"] for d in diagnostics],
            dtype=bool,
        ),
        "n_asset_refits_with_six_start_rescue_attempt_t": np.asarray(
            [d["n_asset_refits_with_six_start_rescue_attempt"] for d in diagnostics],
            dtype=np.int64,
        ),
        "n_asset_refits_rescued_by_six_start_t": np.asarray(
            [d["n_asset_refits_rescued_by_six_start"] for d in diagnostics],
            dtype=np.int64,
        ),
    }
    for key, values in telemetry.items():
        if values.shape != (B,):
            raise RuntimeError(f"unexpected telemetry shape for {key}: {values.shape}")
    return seeds, full, telemetry


# ---------------------------------------------------------------------------
# One spec end to end. spec in {"baseline","crisis","full"}.
# Returns a dict row; also runs the Gaussian comparator for an in-process
# apples-to-apples SD-widening validation (same seeds, same design).
# ---------------------------------------------------------------------------
def run_spec(spec, B, n_jobs, seed, validate_gaussian=True):
    if B <= 0:
        raise ValueError("B must be positive")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    print(f"\n{'='*72}\n=== SPEC: {spec} ===\n{'='*72}")
    if spec == "baseline":
        design, inf_d, reg_d, ret_df = c7.build_design()
        crisis_windows = {a: None for a in ASSETS}
    elif spec == "crisis":
        design, inf_d, reg_d, ret_df, crisis_windows = c8h.build_design_with_regimes("crisis")
    elif spec == "full":
        design, inf_d, reg_d, ret_df, crisis_windows = c8h.build_design_with_regimes("full")
    else:
        raise ValueError(spec)

    for a in ASSETS:
        nreg = design[a].get("n_regime", 0)
        print(f"  {a}: n_obs={design[a]['returns'].shape[0]} n_regime={nreg}")

    print("Fitting observed (unrestricted + null), multistart...")
    t0 = time.time()
    observed = c7.fit_observed(design, seed=seed)
    print(f"  done {time.time()-t0:.1f}s")

    di = np.array([observed[a]["delta_infra"] for a in ASSETS])
    dr = np.array([observed[a]["delta_reg"] for a in ASSETS])
    d_obs = di - dr
    d_bar_obs = float(d_obs.mean())
    multiplier = float(di.mean() / dr.mean())
    print(f"  d_bar_obs={d_bar_obs:.4f}  multiplier={multiplier:.3f}x")

    # correlations + PD cholesky (identical to c7/c8h)
    R_return = ret_df.corr().values
    rho_return = c7.mean_off_diag(R_return)
    z_df = pd.DataFrame({a: pd.Series(observed[a]["z_resid"], index=design[a]["index"])
                         for a in ASSETS}).dropna()
    R_z = z_df.corr().values
    rho_resid = c7.mean_off_diag(R_z)
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
    nu_unr, nu_null = _install_nu(observed)
    audit_payload = _fixed_dgp_audit_payload(spec)
    print(f"  fitted nu (null) = {np.round(nu_null,3)}  median nu_c={np.median(nu_null):.3f}")
    print(
        "  analysis fingerprint="
        f"{str(audit_payload['audit_analysis_stack_sha256'])[:16]}..."
    )
    print(
        f"  union calendar n={c7._GLOBAL['n_calendar']}; availability patterns="
        + ", ".join(
            f"{int(code)}:{int(count)}"
            for code, count in zip(
                c7._GLOBAL["availability_pattern_codes"],
                c7._GLOBAL["availability_pattern_counts"],
            )
        )
    )

    # ---- Student-t-copula null-imposed fixed-path bootstrap --------------------
    print(f"[t-copula NULL-imposed] B={B} n_jobs={n_jobs} (USE_T_COPULA={USE_T_COPULA})...")
    t0 = time.time()
    draw_seed_offset = 10_000
    first_draw_seed = seed + draw_seed_offset
    draw_seeds, null_t_full, t_telemetry = _run_t_bootstrap_full(
        B, n_jobs, first_draw_seed
    )
    # c7's historical contract marks exclusions with NaN; preserve that exact
    # rule so legacy compressed arrays and p-value denominators do not drift.
    usable_t = ~np.isnan(null_t_full)
    null_t = null_t_full[usable_t]
    drop_t = int((~usable_t).sum())
    Bn = len(null_t)
    if Bn == 0:
        raise RuntimeError(f"all {B} t-copula refits were excluded for spec={spec}")
    rescue_attempted_t = t_telemetry["draw_had_six_start_rescue_attempt_t"]
    rescue_retained_t = t_telemetry["draw_retained_after_six_start_rescue_t"]
    asset_rescue_attempts_t = t_telemetry[
        "n_asset_refits_with_six_start_rescue_attempt_t"
    ]
    asset_rescue_successes_t = t_telemetry[
        "n_asset_refits_rescued_by_six_start_t"
    ]
    rescue_failed_t = rescue_attempted_t & (~usable_t)
    p_one_t = (np.sum(null_t >= d_bar_obs) + 1) / (Bn + 1)
    p_two_t = (np.sum(np.abs(null_t) >= abs(d_bar_obs)) + 1) / (Bn + 1)
    sd_t = float(null_t.std())
    print(f"  done {time.time()-t0:.1f}s used={Bn} dropped={drop_t} ({drop_t/B:.1%})")
    print(f"  rescue draws attempted={int(rescue_attempted_t.sum())};"
          f" retained={int(rescue_retained_t.sum())}; failed={int(rescue_failed_t.sum())}")
    print(f"  rescue asset refits attempted={int(asset_rescue_attempts_t.sum())};"
          f" rescued={int(asset_rescue_successes_t.sum())}")
    print(
        f"  t-copula one-sided p={p_one_t:.4f}  "
        f"absolute-statistic p={p_two_t:.4f}"
    )
    print(f"  null d_bar mean={null_t.mean():.4f}  SD={sd_t:.4f}")

    # ---- VALIDATION: run the Gaussian comparator, SAME seeds/design -----------
    p_one_g = sd_g = float("nan")
    drop_g = 0
    null_g = np.array([], dtype=float)
    null_g_full = np.full(B, np.nan, dtype=float)
    usable_g = np.zeros(B, dtype=bool)
    excluded_g = np.zeros(B, dtype=bool)
    if validate_gaussian:
        print("[Gaussian comparator NULL-imposed -- validation, same seeds]...")
        t0 = time.time()
        gaussian_seeds, null_g_full = _run_scalar_bootstrap_full(
            c7._draw_parametric_null, B, n_jobs, first_draw_seed
        )
        if not np.array_equal(gaussian_seeds, draw_seeds):
            raise RuntimeError("t-copula and Gaussian draw-seed arrays differ")
        usable_g = ~np.isnan(null_g_full)
        excluded_g = ~usable_g
        null_g = null_g_full[usable_g]
        drop_g = int(excluded_g.sum())
        Bg = len(null_g)
        if Bg == 0:
            raise RuntimeError(f"all {B} Gaussian refits were excluded for spec={spec}")
        p_one_g = (np.sum(null_g >= d_bar_obs) + 1) / (Bg + 1)
        sd_g = float(null_g.std())
        print(f"  done {time.time()-t0:.1f}s used={Bg} dropped={drop_g} ({drop_g/B:.1%})")
        print(f"  Gaussian p={p_one_g:.4f}  SD={sd_g:.4f}")
        widened = sd_t > sd_g
        print(f"  >>> SD widened (t > Gaussian)?  {widened}  ({sd_t:.4f} vs {sd_g:.4f})")
        print(f"  >>> p rose (t >= Gaussian)?      {p_one_t >= p_one_g}  ({p_one_t:.4f} vs {p_one_g:.4f})")

    # Legacy compressed arrays remain available for c10/c13/c19/c20 and older
    # verifiers.  The full arrays and masks make every draw auditable back to
    # its deterministic seed, including excluded refits.
    draws_path = c2.OUT_DIR / f"c9-tcopula-draws-{spec}.npz"
    draws_tmp = draws_path.with_name(draws_path.stem + ".write-tmp.npz")
    np.savez(
        draws_tmp,
        null_t=null_t,
        null_gaussian=null_g,
        null_t_full=null_t_full,
        null_t_usable_mask=usable_t,
        null_t_excluded_mask=~usable_t,
        null_gaussian_full=null_g_full,
        null_gaussian_usable_mask=usable_g,
        null_gaussian_excluded_mask=excluded_g,
        gaussian_validation_run=np.bool_(validate_gaussian),
        draw_seeds=draw_seeds,
        seed_schema_version=np.int64(2),
        seed_schema=np.asarray(
            "draw_seeds[i] = base_seed + draw_seed_offset + i; "
            "each seed draws one n_calendar-by-6 latent array on the union calendar; "
            "full statistic arrays preserve seed order and use NaN for excluded refits"
        ),
        base_seed=np.int64(seed),
        draw_seed_offset=np.int64(draw_seed_offset),
        first_draw_seed=np.int64(draw_seeds[0]),
        last_draw_seed=np.int64(draw_seeds[-1]),
        B_requested=np.int64(B),
        d_bar_obs=d_bar_obs,
        **t_telemetry,
        **audit_payload,
    )
    draws_tmp.replace(draws_path)

    upper_hits_t = int(np.sum(null_t >= d_bar_obs))
    abs_hits_t = int(np.sum(np.abs(null_t) >= abs(d_bar_obs)))
    lower_hits_t = int(np.sum(null_t <= d_bar_obs))
    upper_hits_g = int(np.sum(null_g >= d_bar_obs)) if validate_gaussian else 0

    return {
        "spec": spec,
        "base_seed": seed,
        "draw_seed_offset": draw_seed_offset,
        "first_draw_seed": int(draw_seeds[0]),
        "last_draw_seed": int(draw_seeds[-1]),
        "B_requested": B,
        "n_calendar": int(c7._GLOBAL["n_calendar"]),
        "availability_pattern_codes": ";".join(
            str(int(v)) for v in c7._GLOBAL["availability_pattern_codes"]
        ),
        "availability_pattern_counts": ";".join(
            str(int(v)) for v in c7._GLOBAL["availability_pattern_counts"]
        ),
        "design_sha256": str(audit_payload["audit_design_sha256"]),
        "fixed_null_dgp_sha256": str(
            audit_payload["audit_fixed_null_dgp_sha256"]
        ),
        "refitter_sha256": str(audit_payload["audit_refitter_sha256"]),
        "simulator_sha256": str(audit_payload["audit_simulator_sha256"]),
        "analysis_stack_sha256": str(
            audit_payload["audit_analysis_stack_sha256"]
        ),
        "multiplier": multiplier,
        "d_bar_obs": d_bar_obs,
        "rho_return": rho_return,
        "rho_resid": rho_resid,
        "nu_null_median": float(np.median(nu_null)),
        "p_gaussian_old_one_sided": float(p_one_g),
        "null_sd_gaussian": sd_g,
        "p_tcopula_one_sided": float(p_one_t),
        "p_tcopula_two_sided": float(p_two_t),
        "null_sd_tcopula": sd_t,
        "sd_widened": bool(sd_t > sd_g) if validate_gaussian else None,
        "B_used_t": Bn,
        "n_dropped_t": drop_t,
        "frac_dropped_t": drop_t / B,
        "n_upper_tail_hits_t": upper_hits_t,
        "n_abs_stat_hits_t": abs_hits_t,
        "n_lower_tail_hits_t": lower_hits_t,
        "n_draws_with_six_start_rescue_attempt_t": int(rescue_attempted_t.sum()),
        "n_draws_retained_after_six_start_rescue_t": int(rescue_retained_t.sum()),
        "n_draws_failed_after_six_start_rescue_t": int(rescue_failed_t.sum()),
        "n_asset_refits_with_six_start_rescue_attempt_t": int(asset_rescue_attempts_t.sum()),
        "n_asset_refits_rescued_by_six_start_t": int(asset_rescue_successes_t.sum()),
        "gaussian_validation_run": bool(validate_gaussian),
        "B_used_gaussian": len(null_g),
        "n_dropped_gaussian": drop_g,
        "frac_dropped_gaussian": drop_g / B if validate_gaussian else np.nan,
        "n_upper_tail_hits_gaussian": upper_hits_g,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=22)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the in-process Gaussian re-run validation")
    args = ap.parse_args()

    t_start = time.time()
    print(f"numba available: {_HAVE_NUMBA}   USE_T_COPULA={USE_T_COPULA}")
    rows = []
    # baseline first (it carries the seed offsets c7 used so p matches its CSV)
    for spec in ["baseline", "crisis", "full"]:
        rows.append(run_spec(spec, args.B, args.n_jobs, args.seed,
                             validate_gaussian=not args.no_validate))

    df = pd.DataFrame(rows)
    out_csv = c2.OUT_DIR / "c9-tcopula-results.csv"
    # Preserve every binary64 scalar on a read/write round trip.  The default
    # pandas formatter can shorten a value to the adjacent float (for example,
    # the baseline add-one p-value), which breaks exact artifact provenance.
    df.to_csv(out_csv, index=False, float_format="%.17g")
    print(f"\nSaved {out_csv}")
    write_finding(rows, args.B, time.time() - t_start)
    print(f"\nTOTAL {(time.time()-t_start)/60:.1f} min")


def _verdict(p):
    if p < 0.05:
        return f"SIGNIFICANT at 5% (p={p:.4f})."
    if p < 0.10:
        return f"MARGINAL: significant at 10% but NOT at 5% (p={p:.4f})."
    return f"NOT significant at 10% -- directional only (p={p:.4f})."


def write_finding(rows, B, elapsed):
    by = {r["spec"]: r for r in rows}
    L = []
    L.append("# C9 -- Student-t-copula CCC-GARCH-X fixed-path bootstrap\n")
    L.append(f"_B={B}; copula={'true multivariate-t (shared chi-square mixing)' if USE_T_COPULA else 'Gaussian copula'} "
             f"with per-asset Student-t margins at the FITTED nu; runtime {elapsed/60:.1f} min._\n")

    L.append("## Motivation\n")
    L.append("c7 and c8h are the cross-asset-robust significance tests for the "
             "infrastructure-vs-regulatory variance-coefficient asymmetry. Their CCC "
             "parametric bootstrap drew the standardised innovations as **Gaussian** "
             "(`rng.standard_normal`) even though each GJR-GARCH-X is fitted with "
             "**Student-t** errors, nu ~ 3.1-4.6. The Gaussian draw therefore does not "
             "match the fitted innovation margins or their joint tail behaviour. The c10 "
             "internal calibration shows material over-rejection for that comparator in "
             "the fitted design; the direction is not asserted as universal.\n")

    L.append("## The fix\n")
    L.append("Innovations are now drawn from a Student-t copula: latent "
             "MVN(0, R_z)" + (" divided by a shared chi-square (-> multivariate-t, joint "
             "tail dependence)" if USE_T_COPULA else "") + ", mapped to uniforms, then to "
             "per-asset Student-t margins at the **fitted nu**, rescaled to unit variance "
             "by sqrt((nu-2)/nu). Same change in the null-imposed and unrestricted draws; "
             "one joint six-vector is generated for every union-calendar date and unavailable "
             "coordinates are discarded. Thus the five pre-BNB assets retain the marginal "
             "dependence implied by R_z. Everything else (B, null via combined dummy, refit, "
             "drop guards) uses the shared c7/c8h engine.\n")

    L.append("## Results: Gaussian comparator vs Student-t specification\n")
    L.append("| spec | multiplier | Gaussian p (1-sided) | **t-copula p (1-sided)** | t-copula absolute-statistic p | null SD Gauss -> t |")
    L.append("|---|---|---|---|---|---|")
    order = ["baseline", "crisis", "full"]
    for spec in order:
        r = by[spec]
        mult = f"{r['multiplier']:.2f}x"
        L.append(f"| {spec} | {mult} | {r['p_gaussian_old_one_sided']:.4f} | "
                 f"**{r['p_tcopula_one_sided']:.4f}** | {r['p_tcopula_two_sided']:.4f} | "
                 f"{r['null_sd_gaussian']:.4f} -> {r['null_sd_tcopula']:.4f} |")
    L.append("")
    L.append("The retained machine label `crisis` denotes one asset-specific "
             "high-variance window from each asset's last 2021 conditional-variance "
             "break to its first break in 2022 or later. Four windows end around FTX; "
             "BNB and ADA end in July 2022, so this is a broad regime control rather "
             "than an FTX-specific control.\n")

    L.append("## Exact draw-retention and rescue telemetry\n")
    L.append("The full NPZ arrays retain all requested seed positions; compressed legacy "
             "arrays contain only usable values. Rescue counts below refer to the single "
             "deterministic six-start retry after a failed or degenerate initial refit.")
    L.append("| spec | t retained / requested | t excluded | rescue draws attempted / retained / failed | asset refits attempted / rescued | Gaussian retained / requested | Gaussian excluded |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for spec in order:
        r = by[spec]
        L.append(
            f"| {spec} | {r['B_used_t']} / {r['B_requested']} | {r['n_dropped_t']} | "
            f"{r['n_draws_with_six_start_rescue_attempt_t']} / "
            f"{r['n_draws_retained_after_six_start_rescue_t']} / "
            f"{r['n_draws_failed_after_six_start_rescue_t']} | "
            f"{r['n_asset_refits_with_six_start_rescue_attempt_t']} / "
            f"{r['n_asset_refits_rescued_by_six_start_t']} | "
            f"{r['B_used_gaussian']} / {r['B_requested']} | "
            f"{r['n_dropped_gaussian']} |"
        )
    L.append("")

    L.append("## Direct comparison under matched seeds\n")
    L.append("The two innovation specifications are rerun with the same seeds and fixed-path "
             "design. A change in p has no predetermined validation direction; the relevant "
             "checks are the intended margins, dependence, and stated calibration results:")
    for spec in order:
        r = by[spec]
        L.append(f"- **{spec}**: p {r['p_gaussian_old_one_sided']:.4f} (Gaussian) -> "
                 f"{r['p_tcopula_one_sided']:.4f} (t-copula); "
                 f"null SD {r['null_sd_gaussian']:.4f} -> {r['null_sd_tcopula']:.4f}.")
    L.append("")
    L.append("**Interpretation.** The fitted Student-t margins are rescaled to unit variance. "
             "The table reports the realised direction and magnitude of each change for these "
             "three fitted specifications. Neither a wider-null heuristic nor a predetermined "
             "direction of the p-value is a general correctness test.")

    L.append("## Per-spec verdict\n")
    for spec in order:
        r = by[spec]
        mult = f"{r['multiplier']:.2f}x"
        L.append(f"- **{spec}** ({mult}, d_bar_obs={r['d_bar_obs']:.3f}): {_verdict(r['p_tcopula_one_sided'])}")
    L.append("")

    L.append("## Honest headline\n")
    pb = by["baseline"]["p_tcopula_one_sided"]
    pc = by["crisis"]["p_tcopula_one_sided"]
    pf = by["full"]["p_tcopula_one_sided"]
    if pb < 0.10:
        head = (f"With the specified Student-t innovations the baseline result is **p={pb:.4f} "
                f"-- still marginal** (sig at 10%, not 5%). The point estimate ({by['baseline']['multiplier']:.2f}x) is "
                f"unchanged; the innovation specification changes the reference distribution.")
    else:
        head = (f"With the specified Student-t innovations the baseline result is **p={pb:.4f} "
                f"-- NOT SIGNIFICANT** (>0.10). The fitted Student-t specification gives a "
                f"higher p than the Gaussian comparator in this design. The point estimate ({by['baseline']['multiplier']:.2f}x) is unchanged; "
                f"the conditional fixed-path inference does not reject. The recursive "
                f"sensitivity is reported separately.")
    L.append(head + "\n")
    L.append("Single-regime-control stability (legacy machine key `crisis`): "
             f"the asset-specific high-variance-regime p is {pc:.4f}, versus "
             f"baseline p={pb:.4f}; full-regime p={pf:.4f}.\n")

    L.append("## Caveats\n")
    L.append("- The copula df is set to the median fitted nu (a single shared "
             "tail-dependence parameter), while the margins use each asset's fitted nu. "
             "C20 reports sensitivity to alternative shared-df values; this choice is "
             "a modelling assumption, not a guarantee of conservatism.")
    L.append("- R_z is calibrated on the six-asset complete-case residual window. Simulation "
             "uses the union calendar: a six-vector and one shared chi-square mixer are drawn "
             "per date, then unavailable asset coordinates are discarded. This uses the "
             "corresponding marginal of R_z on five-asset pre-BNB dates; it extrapolates the "
             "complete-case dependence estimate to that earlier overlap.")
    L.append("- Bootstrap refits use a single default start followed, when necessary, by a "
             "deterministic six-start rescue. Remaining degenerate refits are excluded and "
             "counted. The reported p-values use add-one Monte Carlo smoothing over retained "
             "refits, p=(hits+1)/(B_used+1); the full arrays and "
             "masks disclose the excluded seed positions.")
    L.append("- `USE_T_COPULA` toggles true-t-copula vs Gaussian-copula-with-t-margins; the "
             "headline above uses " + ("the true t-copula." if USE_T_COPULA else "the Gaussian copula."))
    L.append("")
    L.append("## Files\n")
    L.append("- `c9-tcopula-results.csv`, `c9-tcopula-draws-{baseline,crisis,full}.npz` "
             "(legacy compressed arrays plus full seed-indexed arrays, masks, and rescue telemetry)")
    L.append("- `code/c9_tcopula_bootstrap.py` (reuses `c7_ccc_garchx_bootstrap.py` + "
             "`c8h_break_controls_ccc_bootstrap.py` engines)")

    (c2.OUT_DIR / "c9-tcopula-bootstrap-FINDING.md").write_text("\n".join(L))
    print(f"Saved {c2.OUT_DIR / 'c9-tcopula-bootstrap-FINDING.md'}")


if __name__ == "__main__":
    main()
