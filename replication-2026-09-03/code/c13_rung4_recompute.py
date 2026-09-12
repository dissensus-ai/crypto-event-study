"""
C13: RUNG-4 (design-effect) sensitivity audit.
================================================================================

The inference ladder's rung 4 is a diagnostic approximation rather than the
paper's test of record. This script varies two reference-sensitive choices that
materially change that approximation and reports the resulting range instead of
selecting one apparently definitive p-value.

#D1  Effective-df reference sensitivity.
     With DEFF = 1 + (N-1)*rho_bar the *effective* sample size is N_eff = N/DEFF,
     and this diagnostic maps the design effect to
     df_eff = (N-1)/DEFF. That mapping is a heuristic sensitivity convention,
     not a uniquely implied finite-sample distribution or a formal
     Satterthwaite derivation for these six estimators. The script therefore
     reports it alongside the normal and t(N-1) references.

#D4  Dependence-input sensitivity.
     The legacy calculation uses raw-return correlation rho_ret = 0.688. The object
     being averaged is the per-asset SIGNED DIFFERENCE estimator
         d_i = delta_infra,i - delta_reg,i,
     so corr(d_i, d_j) is a more target-aligned alternative to raw-return
     correlation. It is estimated from retained draws of the fitted-null
     t-copula CCC-GARCH-X bootstrap by capturing the PER-ASSET delta draws
     (c7/c9 only stored the aggregated d_bar). It remains conditional on that
     fitted DGP, refit screen, and Monte Carlo sample; it is not a model-free
     population correlation. rho_d_bar is the mean off-diagonal of corr(d).

#MC  Size-study consistency (Table 8 / c10).
     Re-tabulate the design-effect statistic's conditional rejection rates under
     alternative critical-value conventions using the saved c10 fitted-DGP panel
     draws. These rates diagnose calibration under that simulation design; they
     do not establish size under the unknown data-generating process.

Run:
    python c13_rung4_recompute.py --B 2000 --n_jobs 22

Outputs (results/):
    c13-rung4-recompute-results.csv
    c13-rung4-recompute-FINDING.md
    c13-rung4-perasset-draws.npz
"""
import argparse
import hashlib
import inspect
import time
import warnings
import multiprocessing as mp
from multiprocessing import Pool
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
import c9_tcopula_bootstrap as c9
import c10_size_study as c10
from c9_tcopula_bootstrap import _install_nu, USE_T_COPULA
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA

ASSETS = c2.ASSETS
N_STARTS_FIT = 6
MAX_ITER = 2000
DELTA_CAP = c7.DELTA_CAP
C13_ANALYSIS_CONTRACT_VERSION = 2


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_scalar(value):
    array = np.asarray(value)
    if array.shape != ():
        raise RuntimeError(f"expected scalar field, got shape {array.shape}")
    return str(array.item())


# ---------------------------------------------------------------------------
# Per-asset null-imposed t-copula draw that returns the FULL vector of per-asset
# d_i = delta_infra,i - delta_reg,i (not just the mean). The simulator mirrors
# c9._draw_parametric_null_t's fitted-null construction but returns the 6-vector
# instead of its mean. It applies the same one-time six-start rescue and
# degeneracy guard as c7/c9.
# Returns telemetry so the shared optimiser effort is auditable.
# ---------------------------------------------------------------------------
def _draw_perasset_null_t(seed):
    rng = np.random.default_rng(seed)
    design = c7._GLOBAL["design"]
    obs = c7._GLOBAL["observed"]
    nu_vec = c7._GLOBAL["nu_null"]
    nu_c = c7._GLOBAL["nu_c_null"]

    z_by_asset = c9._student_t_calendar_innovations(rng, nu_vec, nu_c)
    d = np.empty(len(ASSETS))
    rescue_attempts = 0
    rescue_successes = 0
    for j, a in enumerate(ASSETS):
        sig2 = obs[a]["sigma2_null"]
        mean_r = obs[a]["mean_return"]
        eps = z_by_asset[a] * np.sqrt(sig2)
        returns = mean_r + eps
        est = FastTARCHX(returns, design[a]["exog_unr"])
        p, f, ok = est.fit(start=None, max_iter=MAX_ITER)
        if not ok or abs(p[5]) > DELTA_CAP or abs(p[6]) > DELTA_CAP:
            rescue_attempts += 1
            p, f, ok = est.fit_multistart(
                n_starts=6,
                seed=c7._GLOBAL.get("rescue_seed", 20260807),
                max_iter=MAX_ITER,
            )
            if not ok or abs(p[5]) > DELTA_CAP or abs(p[6]) > DELTA_CAP:
                return None, rescue_attempts, rescue_successes
            rescue_successes += 1
        d[j] = p[5] - p[6]
    return d, rescue_attempts, rescue_successes


def run_perasset_bootstrap(B, n_jobs, base_seed, return_full=False):
    """Run per-asset draws, optionally retaining exact seed-indexed arrays."""
    seeds = np.arange(base_seed, base_seed + B, dtype=np.int64)
    with Pool(processes=n_jobs) as pool:
        res = pool.map(
            _draw_perasset_null_t,
            seeds.tolist(),
            chunksize=max(1, B // (n_jobs * 4)),
        )
    D_full = np.full((B, len(ASSETS)), np.nan, dtype=float)
    rescue_attempts_by_draw = np.zeros(B, dtype=np.int64)
    rescue_successes_by_draw = np.zeros(B, dtype=np.int64)
    for i, (d, attempts, successes) in enumerate(res):
        rescue_attempts_by_draw[i] = attempts
        rescue_successes_by_draw[i] = successes
        if d is not None:
            D_full[i] = d
    usable_mask = ~np.isnan(D_full).any(axis=1)
    kept = D_full[usable_mask]
    n_drop = int((~usable_mask).sum())
    D = np.asarray(kept, dtype=float)  # (B_used, 6)
    n_draws_with_rescue = sum(int(attempts > 0) for _d, attempts, _successes in res)
    n_asset_rescue_attempts = sum(attempts for _d, attempts, _successes in res)
    n_asset_rescue_successes = sum(successes for _d, _attempts, successes in res)
    legacy = (
        D, n_drop, n_draws_with_rescue,
        n_asset_rescue_attempts, n_asset_rescue_successes,
    )
    if not return_full:
        return legacy
    return legacy + (
        seeds,
        D_full,
        usable_mask,
        rescue_attempts_by_draw,
        rescue_successes_by_draw,
    )


def _assert_exact_c9_alignment(args, d_obs, D_full, usable_mask, local_audit):
    """Assert c13 reproduces c9's baseline statistic at every paired seed."""
    path = c2.OUT_DIR / "c9-tcopula-draws-baseline.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required; rerun c9 baseline before the final c13 run"
        )
    with np.load(path) as archive:
        required = {
            "null_t", "null_t_full", "null_t_usable_mask", "draw_seeds",
            "B_requested", "base_seed", "draw_seed_offset", "d_bar_obs",
            "seed_schema_version", "audit_design_sha256",
            "audit_fixed_null_dgp_sha256", "audit_refitter_sha256",
            "audit_simulator_sha256",
            "audit_analysis_stack_sha256",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"c9 baseline lacks audit fields: {sorted(missing)}")
        fixed = {key: archive[key] for key in required}

    expected_seeds = np.arange(
        args.seed + 10_000, args.seed + 10_000 + args.B, dtype=np.int64
    )
    if int(fixed["seed_schema_version"]) != 2:
        raise RuntimeError("c9 baseline does not use union-calendar seed schema v2")
    if int(fixed["B_requested"]) != args.B:
        raise RuntimeError("c9/c13 requested draw counts differ")
    if int(fixed["base_seed"]) != args.seed or int(fixed["draw_seed_offset"]) != 10_000:
        raise RuntimeError("c9/c13 base seed or draw-seed offset differs")
    if not np.array_equal(np.asarray(fixed["draw_seeds"]), expected_seeds):
        raise RuntimeError("c9/c13 draw seed arrays differ")
    if float(fixed["d_bar_obs"]) != float(np.mean(d_obs)):
        raise RuntimeError("c9/c13 observed d_bar differs exactly")
    for key in (
        "audit_design_sha256",
        "audit_fixed_null_dgp_sha256",
        "audit_refitter_sha256",
        "audit_simulator_sha256",
        "audit_analysis_stack_sha256",
    ):
        if str(fixed[key]) != str(local_audit[key]):
            raise RuntimeError(f"c9/c13 {key} differs")

    c9_full = np.asarray(fixed["null_t_full"], dtype=float)
    c9_usable = np.asarray(fixed["null_t_usable_mask"], dtype=bool)
    c13_full = np.mean(D_full, axis=1)
    if not np.array_equal(c9_usable, usable_mask):
        mismatch = np.flatnonzero(c9_usable != usable_mask)
        raise RuntimeError(
            f"c9/c13 retained masks differ at {len(mismatch)} seeds; first={mismatch[:5]}"
        )
    if not np.array_equal(c9_full, c13_full, equal_nan=True):
        mismatch = np.flatnonzero(~np.isclose(
            c9_full, c13_full, rtol=0.0, atol=0.0, equal_nan=True
        ))
        raise RuntimeError(
            f"c9/c13 per-seed d_bar differs exactly at {len(mismatch)} seeds; "
            f"first={mismatch[:5]}"
        )
    if not np.array_equal(np.asarray(fixed["null_t"]), c13_full[usable_mask]):
        raise RuntimeError("c9 legacy null_t differs from c13's usable compression")
    return c13_full


# ---------------------------------------------------------------------------
def deff_df_p(mean_d, se_naive, N, rho):
    """
    Design-effect sensitivity calculations for a given correlation rho.
      DEFF   = 1 + (N-1)*rho
      N_eff  = N / DEFF             (design-effect-equivalent label)
      df_eff = (N-1)/DEFF        (effective-df sensitivity convention)
      se_de  = se_naive * sqrt(DEFF)
      t      = mean_d / se_de    (statistic from the dispersion of the 6 diffs)
    Returns dict with DEFF, N_eff, df_eff, se_de, t, and ONE-SIDED p under both:
      p_t1   = t(df_eff) reference   (effective-df sensitivity)
      p_t5   = t(N-1) reference      (c6's current rung-4 df convention)
      p_norm = normal reference      (the ladder-narrative rung-4 tail)
    No reference in this function is asserted to be an exact finite-sample law.
    """
    DEFF = 1.0 + (N - 1) * rho
    N_eff = N / DEFF
    df_eff = (N - 1) / DEFF
    se_de = se_naive * np.sqrt(DEFF)
    t = mean_d / se_de
    return {
        "rho": rho, "DEFF": DEFF, "N_eff": N_eff, "df_eff": df_eff,
        "se_de": se_de, "t_dispersion": t,
        "p_t_dfeff_dispersion": float(stats.t.sf(t, df=df_eff)),
        "p_t_N1_dispersion": float(stats.t.sf(t, df=N - 1)),
        "p_norm_dispersion": float(stats.norm.sf(t)),
    }


def _validate_current_c10_archive(npz_path, c9_analysis_stack_sha256):
    """Reject a c10 input not produced by the current c10 decision contract."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required; run the final c10 study before c13"
        )
    required = {
        "p_de", "c9_exact_identity_verified",
        "c10_analysis_contract_version", "c10_analysis_contract",
        "c10_analysis_sha256", "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(
                f"{path.name} lacks current c10 provenance: {sorted(missing)}"
            )
        saved = {key: archive[key] for key in required}
    if not bool(saved["c9_exact_identity_verified"]):
        raise RuntimeError("c10 archive does not certify exact c9 identity")
    current = c10.current_c10_analysis_provenance(c9_analysis_stack_sha256)
    if int(saved["c10_analysis_contract_version"]) != int(
        current["c10_analysis_contract_version"]
    ):
        raise RuntimeError("c10 analysis-contract version differs from current source")
    for key in (
        "c10_analysis_contract",
        "c10_analysis_sha256",
        "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    ):
        if _text_scalar(saved[key]) != str(current[key]):
            raise RuntimeError(f"c10 {key} differs from current source")
    return {
        "path": f"results/{path.name}",
        "archive_sha256": _sha256_file(path),
        **current,
    }


# ---------------------------------------------------------------------------
def mc_size_under_t_crit(npz_path, N, c9_analysis_stack_sha256,
                         rho_for_crit_label="return"):
    """
    #MC: re-tabulate the design-effect statistic on saved c10 per-panel draws
    under an alternative t(df_eff) one-sided critical value. The per-panel
    design-effect p-values saved by c10 (p_de) used stats.t.sf(t_ii, df=N-1),
    with df=N-1=5. We recover each panel's statistic by inverting that monotone
    transform (t = t.isf(p_de, df=N-1)) and apply each listed reference.

    We report conditional fitted-DGP rejection rates at nominal 0.05/0.10 under:
      - normal (z one-sided 1.645 / 1.282)  -- what the headline 'design-effect'
        narrative rung implies
      - t(N-1=5)                              -- c10's actual saved rule
      - t(df_eff)                             -- effective-df sensitivity
    These are reference comparisons, not proof that any one reference has the
    exact sampling law for the six fitted estimators.
    """
    source_c10 = _validate_current_c10_archive(
        npz_path, c9_analysis_stack_sha256
    )
    with np.load(npz_path, allow_pickle=False) as archive:
        p_de = np.asarray(archive["p_de"], dtype=float)
    # per-panel one-sided p under t(N-1)
    p_de = p_de[np.isfinite(p_de)]
    n = len(p_de)
    # invert to the t-statistic each panel produced (df = N-1, as c10 used)
    t_panel = stats.t.isf(p_de, df=N - 1)  # one-sided upper-tail inverse

    return p_de, t_panel, n, source_c10


def current_c13_analysis_provenance(c9_analysis_stack_sha256,
                                    c10_analysis_stack_sha256):
    """Bind c13's per-asset draws and reference-sensitivity logic to source."""
    contract = (
        f"c13-analysis-contract-v{C13_ANALYSIS_CONTRACT_VERSION}\n"
        f"N_STARTS_FIT={N_STARTS_FIT}\n"
        f"MAX_ITER={MAX_ITER}\n"
        f"DELTA_CAP={DELTA_CAP:.17g}\n"
        f"ASSETS={','.join(ASSETS)}\n"
        + "\n".join(inspect.getsource(function) for function in (
            _draw_perasset_null_t,
            run_perasset_bootstrap,
            _assert_exact_c9_alignment,
            deff_df_p,
            _validate_current_c10_archive,
            mc_size_under_t_crit,
            main,
        ))
    )
    analysis_sha256 = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    source_sha256 = _sha256_file(Path(__file__).resolve())
    parent_c9 = str(c9_analysis_stack_sha256)
    parent_c10 = str(c10_analysis_stack_sha256)
    stack_sha256 = hashlib.sha256(
        (
            f"c13-downstream-analysis-stack-v{C13_ANALYSIS_CONTRACT_VERSION}\n"
            f"{parent_c9}\n{parent_c10}\n{analysis_sha256}\n{source_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "c13_analysis_contract_version": C13_ANALYSIS_CONTRACT_VERSION,
        "c13_analysis_contract": contract,
        "c13_analysis_sha256": analysis_sha256,
        "c13_source_sha256": source_sha256,
        "c13_parent_c9_analysis_stack_sha256": parent_c9,
        "c13_parent_c10_analysis_stack_sha256": parent_c10,
        "c13_downstream_analysis_stack_sha256": stack_sha256,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=22)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    t_start = time.time()
    print(f"numba available: {_HAVE_NUMBA}")
    print("Building design (baseline S1 50-event)...")
    design, inf_d, reg_d, ret_df = c7.build_design()
    print(f"  events: {len(inf_d)} infra, {len(reg_d)} reg")

    print("Fitting observed (unrestricted + null), multistart...")
    observed = c7.fit_observed(design, seed=args.seed)

    di = np.array([observed[a]["delta_infra"] for a in ASSETS])
    dr = np.array([observed[a]["delta_reg"] for a in ASSETS])
    d_obs = di - dr
    N = len(d_obs)
    mean_d = float(d_obs.mean())
    se_naive = float(d_obs.std(ddof=1) / np.sqrt(N))
    multiplier = float(di.mean() / dr.mean())
    print(f"  per-asset d_i = {np.round(d_obs,4)}")
    print(f"  mean_d={mean_d:.4f}  se_naive={se_naive:.4f}  multiplier={multiplier:.3f}x")

    # correlations + cholesky of R_z + globals (identical to c7/c9 setup)
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
            R_z_pd = R_z + eps_jit * np.eye(N)
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
    _install_nu(observed)
    local_audit = c9._fixed_dgp_audit_payload("baseline")
    c10_npz_path = c2.OUT_DIR / "c10-size-study-draws.npz"
    source_c10 = _validate_current_c10_archive(
        c10_npz_path, str(local_audit["audit_analysis_stack_sha256"])
    )
    c13_provenance = current_c13_analysis_provenance(
        str(local_audit["audit_analysis_stack_sha256"]),
        source_c10["c10_downstream_analysis_stack_sha256"],
    )
    c9_npz_path = c2.OUT_DIR / "c9-tcopula-draws-baseline.npz"
    if not c9_npz_path.is_file():
        raise FileNotFoundError(
            f"{c9_npz_path} is required; rerun c9 baseline before c13"
        )
    source_c9_archive_sha256 = _sha256_file(c9_npz_path)
    print(f"  rho_return={rho_return:.4f}  rho_resid={rho_resid:.4f}  USE_T_COPULA={USE_T_COPULA}")
    print(
        f"  union calendar n={c7._GLOBAL['n_calendar']}; analysis fingerprint="
        f"{str(local_audit['audit_analysis_stack_sha256'])[:16]}..."
    )
    print(
        "  current c10 source/decision contract PASS; c13 stack="
        f"{c13_provenance['c13_downstream_analysis_stack_sha256'][:16]}..."
    )

    # ---- #D4: fitted-null estimate of corr(d_i, d_j) for sensitivity -------
    print(f"\n[#D4] fitted-null t-copula bootstrap B={args.B} for corr(d_i,d_j) sensitivity...")
    t0 = time.time()
    (D, n_drop, n_draws_with_rescue, n_asset_rescue_attempts,
     n_asset_rescue_successes, draw_seeds, D_full, usable_mask,
     rescue_attempts_by_draw, rescue_successes_by_draw) = run_perasset_bootstrap(
        args.B, args.n_jobs, args.seed + 10_000, return_full=True
    )
    dbar_full = _assert_exact_c9_alignment(
        args, d_obs, D_full, usable_mask, local_audit
    )
    B_used = D.shape[0]
    print(f"  done {time.time()-t0:.1f}s  used={B_used} dropped={n_drop} ({n_drop/args.B:.1%})")
    print(f"  rescue draws={n_draws_with_rescue}; asset attempts={n_asset_rescue_attempts};"
          f" successes={n_asset_rescue_successes}")
    R_d = np.corrcoef(D, rowvar=False)     # 6x6 correlation of the per-asset diffs
    rho_d_bar = c7.mean_off_diag(R_d)
    # Covariance-ratio analogue of a design effect for the equal-weight mean in
    # these retained fitted-null draws. This is algebraically determined by the
    # estimated covariance matrix; it is not an exact population design effect.
    Cov_d = np.cov(D, rowvar=False)
    var_mean_correlated = Cov_d.sum() / N**2
    var_mean_iid = np.trace(Cov_d) / N**2
    DEFF_cov = var_mean_correlated / var_mean_iid
    print(f"  rho_d_bar (mean off-diag corr of d_i) = {rho_d_bar:.4f}")
    print(f"  DEFF (Kish, rho_d_bar)               = {1+(N-1)*rho_d_bar:.4f}")
    print(f"  DEFF_cov (retained-draw covariance ratio) = {DEFF_cov:.4f}")

    if _sha256_file(c9_npz_path) != source_c9_archive_sha256:
        raise RuntimeError("source c9 baseline archive changed during c13 bootstrap")
    source_c10_mid = _validate_current_c10_archive(
        c10_npz_path, str(local_audit["audit_analysis_stack_sha256"])
    )
    if source_c10_mid != source_c10:
        raise RuntimeError("source c10 archive/source changed during c13 bootstrap")
    c13_provenance_end = current_c13_analysis_provenance(
        str(local_audit["audit_analysis_stack_sha256"]),
        source_c10["c10_downstream_analysis_stack_sha256"],
    )
    if c13_provenance_end != c13_provenance:
        raise RuntimeError(
            "c13 source/analysis contract changed during computation; rerun cleanly"
        )

    c13_npz_path = c2.OUT_DIR / "c13-rung4-perasset-draws.npz"
    np.savez(c13_npz_path,
             D=D, D_full=D_full, dbar_full=dbar_full,
             usable_mask=usable_mask, excluded_mask=~usable_mask,
             draw_seeds=draw_seeds,
             rescue_attempts_by_draw=rescue_attempts_by_draw,
             rescue_successes_by_draw=rescue_successes_by_draw,
             R_d=R_d, Cov_d=Cov_d, d_obs=d_obs,
             rho_return=rho_return, rho_resid=rho_resid, rho_d_bar=rho_d_bar,
             base_seed=np.int64(args.seed), first_draw_seed=np.int64(args.seed + 10_000),
             last_draw_seed=np.int64(args.seed + 10_000 + args.B - 1),
             seed_schema_version=np.int64(2),
             exact_c9_baseline_alignment=np.bool_(True),
             audit_design_sha256=local_audit["audit_design_sha256"],
             audit_fixed_null_dgp_sha256=local_audit["audit_fixed_null_dgp_sha256"],
             audit_refitter_sha256=local_audit["audit_refitter_sha256"],
             audit_simulator_sha256=local_audit["audit_simulator_sha256"],
             audit_analysis_stack_sha256=local_audit["audit_analysis_stack_sha256"],
             audit_simulation_calendar_ns=local_audit["audit_simulation_calendar_ns"],
             audit_calendar_pos_concat=local_audit["audit_calendar_pos_concat"],
             audit_calendar_pos_offsets=local_audit["audit_calendar_pos_offsets"],
             audit_availability_pattern_codes=local_audit["audit_availability_pattern_codes"],
             audit_availability_pattern_counts=local_audit["audit_availability_pattern_counts"],
             source_c9_baseline_archive=np.asarray(
                 "results/c9-tcopula-draws-baseline.npz"
             ),
             source_c9_baseline_archive_sha256=np.asarray(
                 source_c9_archive_sha256
             ),
             source_c10_archive=np.asarray(source_c10["path"]),
             source_c10_archive_sha256=np.asarray(source_c10["archive_sha256"]),
             source_c10_analysis_contract_version=np.int64(
                 source_c10["c10_analysis_contract_version"]
             ),
             source_c10_analysis_sha256=np.asarray(
                 source_c10["c10_analysis_sha256"]
             ),
             source_c10_source_sha256=np.asarray(
                 source_c10["c10_source_sha256"]
             ),
             source_c10_downstream_analysis_stack_sha256=np.asarray(
                 source_c10["c10_downstream_analysis_stack_sha256"]
             ),
             c13_analysis_contract_version=np.int64(
                 c13_provenance["c13_analysis_contract_version"]
             ),
             c13_analysis_contract=np.asarray(
                 c13_provenance["c13_analysis_contract"]
             ),
             c13_analysis_sha256=np.asarray(
                 c13_provenance["c13_analysis_sha256"]
             ),
             c13_source_sha256=np.asarray(c13_provenance["c13_source_sha256"]),
             c13_parent_c9_analysis_stack_sha256=np.asarray(
                 c13_provenance["c13_parent_c9_analysis_stack_sha256"]
             ),
             c13_parent_c10_analysis_stack_sha256=np.asarray(
                 c13_provenance["c13_parent_c10_analysis_stack_sha256"]
             ),
             c13_downstream_analysis_stack_sha256=np.asarray(
                 c13_provenance["c13_downstream_analysis_stack_sha256"]
             ),
             n_draws_with_six_start_rescue=np.int64(n_draws_with_rescue),
             n_asset_rescue_attempts=np.int64(n_asset_rescue_attempts),
             n_asset_rescue_successes=np.int64(n_asset_rescue_successes),
             n_post_rescue_exclusions=np.int64(n_drop))
    with np.load(c13_npz_path, allow_pickle=False) as trial:
        if _text_scalar(trial["c13_analysis_contract"]) != str(
            c13_provenance["c13_analysis_contract"]
        ):
            raise RuntimeError("c13 archive changed the analysis contract")
        for key in (
            "c13_analysis_sha256",
            "c13_source_sha256",
            "c13_parent_c9_analysis_stack_sha256",
            "c13_parent_c10_analysis_stack_sha256",
            "c13_downstream_analysis_stack_sha256",
        ):
            if _text_scalar(trial[key]) != str(c13_provenance[key]):
                raise RuntimeError(f"c13 archive changed {key}")
        if _text_scalar(trial["source_c10_archive_sha256"]) != str(
            source_c10["archive_sha256"]
        ):
            raise RuntimeError("c13 archive changed the source-c10 archive hash")

    # ---- #D1 + #D4: build the rung-4 sensitivity table ----------------------
    rho_grid = {
        "rho_return (c6 legacy input)": rho_return,
        "rho_resid (standardized-residual sensitivity)": rho_resid,
        "rho_d_bar (fitted-null estimator-difference sensitivity)": rho_d_bar,
    }
    rows = []
    for label, rho in rho_grid.items():
        r = deff_df_p(mean_d, se_naive, N, rho)
        r["rho_label"] = label
        rows.append(r)
    res_df = pd.DataFrame(rows)

    print("\n=== RUNG-4 SENSITIVITY (dispersion statistic mean_d/se_de) ===")
    print(f"{'rho input':<48}{'rho':>7}{'DEFF':>7}{'df_eff':>8}{'t':>7}"
          f"{'p t(dfeff)':>12}{'p t(5)':>9}{'p norm':>9}")
    for _, r in res_df.iterrows():
        print(f"{r['rho_label']:<48}{r['rho']:>7.3f}{r['DEFF']:>7.2f}{r['df_eff']:>8.2f}"
              f"{r['t_dispersion']:>7.3f}{r['p_t_dfeff_dispersion']:>12.4f}"
              f"{r['p_t_N1_dispersion']:>9.4f}{r['p_norm_dispersion']:>9.4f}")

    # ---- #MC: conditional rejection rates under alternative references ------
    print("\n[#MC] c10 fitted-DGP rejection rates under alternative references...")
    p_de_panel, t_panel, n_panel, source_c10_recheck = mc_size_under_t_crit(
        c10_npz_path,
        N,
        str(local_audit["audit_analysis_stack_sha256"]),
    )
    if source_c10_recheck != source_c10:
        raise RuntimeError("c10 archive changed during the c13 run")
    # df_eff for the size study uses the SAME rho the size study DGP used.
    # c10 design-effect rung used rho_return. We report sensitivity results
    # under df_eff(rho_return), df_eff(rho_d_bar), normal, and t(5).
    df_eff_ret = (N - 1) / (1 + (N - 1) * rho_return)
    df_eff_d = (N - 1) / (1 + (N - 1) * rho_d_bar)
    crit = {
        "normal z (size study's z=1.645)": {
            0.05: stats.norm.isf(0.05), 0.10: stats.norm.isf(0.10)
        },
        "t(N-1=5) [c10 saved rule]": {
            0.05: stats.t.isf(0.05, N - 1), 0.10: stats.t.isf(0.10, N - 1)
        },
        f"t(df_eff={df_eff_ret:.2f}) [sensitivity, rho_return]": {
            0.05: stats.t.isf(0.05, df_eff_ret),
            0.10: stats.t.isf(0.10, df_eff_ret),
        },
        f"t(df_eff={df_eff_d:.2f}) [sensitivity, rho_d_bar]": {
            0.05: stats.t.isf(0.05, df_eff_d),
            0.10: stats.t.isf(0.10, df_eff_d),
        },
    }
    mc_rows = []
    for clabel, cc in crit.items():
        s5 = float(np.mean(t_panel > cc[0.05]))
        s10 = float(np.mean(t_panel > cc[0.10]))
        mc_rows.append({
            "crit_label": clabel,
            "crit_005": cc[0.05],
            "crit_010": cc[0.10],
            "size_005": s5,
            "size_010": s10,
            "n_panels": n_panel,
        })
        print(
            f"  {clabel:<46} crit05={cc[0.05]:7.3f} "
            f"crit10={cc[0.10]:7.3f} size@05={s5:.3f} size@10={s10:.3f}"
        )
    mc = pd.DataFrame(mc_rows)
    print(
        f"\n  [ref] one-sided t_{{0.95,1}}={stats.t.isf(0.05,1):.3f}, "
        f"two-sided t_{{0.975,1}}={stats.t.isf(0.025,1):.3f} "
        "(the 12.706 the prompt cites)"
    )

    if _sha256_file(c9_npz_path) != source_c9_archive_sha256:
        raise RuntimeError("source c9 baseline archive changed during c13 analysis")
    if current_c13_analysis_provenance(
        str(local_audit["audit_analysis_stack_sha256"]),
        source_c10["c10_downstream_analysis_stack_sha256"],
    ) != c13_provenance:
        raise RuntimeError("c13 source/analysis contract changed before output commit")

    # ---- save ---------------------------------------------------------------
    out_csv = c2.OUT_DIR / "c13-rung4-recompute-results.csv"
    with open(out_csv, "w") as f:
        f.write("# C13 rung-4 diagnostic: reference- and dependence-input sensitivity\n")
        f.write(f"# mean_d={mean_d:.6f} se_naive={se_naive:.6f} N={N} multiplier={multiplier:.4f}\n")
        f.write(f"# rho_return={rho_return:.6f} rho_resid={rho_resid:.6f} rho_d_bar={rho_d_bar:.6f}\n")
        f.write(f"# DEFF_cov(retained-draw covariance ratio)={DEFF_cov:.6f}  B_used={B_used} dropped={n_drop}\n")
        f.write(f"# one-rescue parity: rescue_draws={n_draws_with_rescue} "
                f"asset_attempts={n_asset_rescue_attempts} "
                f"asset_successes={n_asset_rescue_successes}\n")
        f.write(
            "# c13_analysis_sha256="
            f"{c13_provenance['c13_analysis_sha256']}\n"
        )
        f.write(
            "# c13_source_sha256="
            f"{c13_provenance['c13_source_sha256']}\n"
        )
        f.write(
            "# c13_downstream_analysis_stack_sha256="
            f"{c13_provenance['c13_downstream_analysis_stack_sha256']}\n"
        )
        f.write(
            "# source_c10_archive_sha256="
            f"{source_c10['archive_sha256']}\n"
        )
        f.write("\n# rung-4 dispersion statistic under alternative rho inputs and references\n")
        res_df.to_csv(f, index=False)
        if mc is not None:
            f.write("\n# #MC: fitted-DGP rejection rates on c10 draws under each reference\n")
            mc.to_csv(f, index=False)
    print(f"\nSaved {out_csv}")

    write_finding(dict(
        mean_d=mean_d, se_naive=se_naive, N=N, multiplier=multiplier,
        rho_return=rho_return, rho_resid=rho_resid, rho_d_bar=rho_d_bar,
        DEFF_cov=DEFF_cov, B_used=B_used, n_drop=n_drop, B=args.B,
        n_draws_with_rescue=n_draws_with_rescue,
        n_asset_rescue_attempts=n_asset_rescue_attempts,
        n_asset_rescue_successes=n_asset_rescue_successes,
        d_obs=d_obs, R_d=R_d,
    ), res_df, mc, time.time() - t_start)
    print(f"\nTOTAL {(time.time()-t_start)/60:.1f} min")


def write_finding(s, res_df, mc, elapsed):
    N = s["N"]
    rr = res_df.set_index("rho_label")
    key_d = [k for k in rr.index if "rho_d_bar" in k][0]
    key_ret = [k for k in rr.index if "rho_return" in k][0]
    cd = rr.loc[key_d]
    cret = rr.loc[key_ret]

    p_sensitivity = res_df["p_t_dfeff_dispersion"].to_numpy(dtype=float)
    p_low = float(np.nanmin(p_sensitivity))
    p_high = float(np.nanmax(p_sensitivity))

    def display_rho_label(label):
        if "rho_d_bar" in label:
            return "rho_d_bar (fitted-null estimator-difference sensitivity)"
        if "rho_return" in label:
            return "rho_return (c6 legacy input)"
        if "rho_resid" in label:
            return "rho_resid (standardized-residual sensitivity)"
        return label

    def display_crit_label(label):
        return label.replace("[corrected,", "[sensitivity,")

    L = []
    L.append("# C13 -- Rung-4 design-effect sensitivity diagnostic\n")
    L.append(
        f"_Per-asset fitted-null t-copula bootstrap: requested B={s['B']}, "
        f"retained {s['B_used']}, excluded {s['n_drop']}; numba={_HAVE_NUMBA}; "
        f"runtime {elapsed/60:.1f} min._\n"
    )

    L.append("## Inferential status\n")
    L.append(
        "This exercise is a heuristic diagnostic, not the paper's test of record "
        "and not a source of a replacement p-value. It asks how the analytical "
        "rung changes when its dependence input and tail reference are varied. "
        "No convention below is asserted to be the exact finite-sample law of six "
        "estimated GARCH-X contrasts."
    )
    L.append(
        "The mapping df_eff=(N-1)/DEFF is an effective-df sensitivity convention. "
        "A design effect does not by itself prove that the resulting statistic is "
        "Student-t with that many degrees of freedom. Likewise, rho_d_bar is "
        "estimated from retained draws under the fitted-null c9 simulator, so it "
        "is more closely aligned with the averaged estimator than raw-return "
        "correlation but remains model-, screen-, seed-, and Monte-Carlo-dependent.\n"
    )

    L.append("## Inputs and retained-draw diagnostics\n")
    L.append(f"- per-asset d_i = {np.round(s['d_obs'], 4).tolist()}")
    L.append(
        f"- observed mean_d = {s['mean_d']:.4f}; six-contrast dispersion SE = "
        f"{s['se_naive']:.4f}; variance-increment ratio = {s['multiplier']:.3f}x"
    )
    L.append(
        f"- rho_return = {s['rho_return']:.4f}; rho_resid = "
        f"{s['rho_resid']:.4f}; fitted-null rho_d_bar = {s['rho_d_bar']:.4f}"
    )
    L.append(
        f"- retained-draw covariance ratio DEFF_cov = {s['DEFF_cov']:.3f}; "
        f"equicorrelation formula at rho_d_bar = "
        f"{1 + (N - 1) * s['rho_d_bar']:.3f}"
    )
    L.append(
        f"- draws invoking the one-rescue rule = {s['n_draws_with_rescue']}; "
        f"asset-level rescue attempts/successes = "
        f"{s['n_asset_rescue_attempts']}/{s['n_asset_rescue_successes']}"
    )
    L.append("")
    L.append(
        "DEFF_cov is an algebraic variance ratio computed from the estimated "
        "covariance matrix of retained bootstrap draws. Calling it a covariance "
        "ratio avoids implying an exact population design effect.\n"
    )

    L.append("### Correlation matrix of fitted-null per-asset difference estimates\n")
    L.append("| | " + " | ".join(ASSETS) + " |")
    L.append("|" + "---|" * (N + 1))
    for i, asset in enumerate(ASSETS):
        L.append(
            f"| {asset} | "
            + " | ".join(f"{s['R_d'][i, j]:.3f}" for j in range(N))
            + " |"
        )
    L.append("")

    L.append("## Analytical reference sensitivity\n")
    L.append(
        "All tail areas are one-sided for the pre-specified infrastructure-greater-"
        "than-regulatory direction. Each row applies the same formulas to a "
        "different dependence proxy; the columns then vary the tail reference.\n"
    )
    L.append(
        "| dependence input | rho | DEFF | N_eff | df_eff convention | t "
        "(dispersion) | p: t(df_eff) | p: t(N-1) | p: normal |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in res_df.iterrows():
        L.append(
            f"| {display_rho_label(row['rho_label'])} | {row['rho']:.3f} | "
            f"{row['DEFF']:.2f} | "
            f"{row['N_eff']:.2f} | {row['df_eff']:.2f} | "
            f"{row['t_dispersion']:.2f} | "
            f"{row['p_t_dfeff_dispersion']:.4f} | "
            f"{row['p_t_N1_dispersion']:.4f} | "
            f"{row['p_norm_dispersion']:.4f} |"
        )
    L.append("")
    L.append(
        f"Across the three dependence inputs for the dispersion statistic under "
        f"the t(df_eff) "
        f"convention, the descriptive p-value span is {p_low:.3f}-{p_high:.3f}. "
        f"For the dispersion statistic alone, changing rho_return "
        f"({cret['rho']:.3f}) to fitted-null rho_d_bar ({cd['rho']:.3f}) changes "
        f"DEFF from {cret['DEFF']:.2f} to {cd['DEFF']:.2f} and the corresponding "
        f"tail area from {cret['p_t_dfeff_dispersion']:.3f} to "
        f"{cd['p_t_dfeff_dispersion']:.3f}. This span is a sensitivity range, "
        f"not a confidence interval and not a multiple-testing adjustment.\n"
    )

    if mc is not None:
        L.append("## Conditional rejection-rate diagnostic using c10 panels\n")
        L.append(
            "The saved c10 panel statistic is recovered by inverting its t(N-1) "
            "tail area and is then compared with alternative critical values. "
            "The resulting rates are conditional on the c10 fitted DGP, retained "
            "panels, refit screen, and simulation settings; they are not estimates "
            "of unconditional size under the unknown true DGP.\n"
        )
        L.append(
            "| reference convention | critical value at .05 | critical value at "
            ".10 | conditional rejection rate at .05 | conditional rejection "
            "rate at .10 | retained panels |"
        )
        L.append("|---|---:|---:|---:|---:|---:|")
        for _, row in mc.iterrows():
            L.append(
                f"| {display_crit_label(row['crit_label'])} | "
                f"{row['crit_005']:.3f} | "
                f"{row['crit_010']:.3f} | {row['size_005']:.3f} | "
                f"{row['size_010']:.3f} | {int(row['n_panels'])} |"
            )
        L.append("")
        L.append(
            "Large movement across these rows demonstrates reference sensitivity. "
            "It does not identify one row as calibrated outside the fitted-DGP "
            "experiment, and it does not independently validate the fixed-path or "
            "recursive bootstrap.\n"
        )

    L.append("## Interpretation\n")
    L.append(
        "The defensible conclusion is narrow: the closed-form rung is highly "
        "sensitive to choices that are difficult to justify with only six fitted "
        "asset contrasts. Raw-return correlation is a legacy proxy; fitted-null "
        "corr(d_i,d_j) is more target-aligned but simulation-conditional. The "
        "effective-df tail is a heavier-tailed reference experiment, not a derived "
        "sampling theorem. Accordingly, rung 4 should remain a diagnostic range "
        "and should not be used to select significance or to certify the paper's "
        "bootstrap inference."
    )
    L.append("")
    L.append(
        "This diagnostic neither proves that the raw-return input is invalid nor "
        "that the estimator-difference input is true. It also does not show that "
        "one innovation law is correct for the unknown DGP. Those questions are "
        "addressed, only conditionally, by the separately reported fitted-DGP "
        "calibration and scheme-sensitivity analyses.\n"
    )

    L.append("## Files\n")
    L.append(
        "- `c13-rung4-recompute-results.csv` -- analytical sensitivity table "
        "and conditional c10 rejection-rate table"
    )
    L.append(
        "- `c13-rung4-perasset-draws.npz` -- retained per-asset difference "
        "draws and their estimated correlation/covariance matrices"
    )
    L.append(
        "- `code/c13_rung4_recompute.py` -- diagnostic generator using the "
        "c7/c9 fitted-null engine and saved c10 draws"
    )

    out = c2.OUT_DIR / "c13-rung4-recompute-FINDING.md"
    out.write_text("\n".join(L) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
