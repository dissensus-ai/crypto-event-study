"""
C20: paired copula-df sensitivity for the fixed-path c9 baseline.

The fitted-median baseline and the two alternative copula-df values are run with
one common observed fit and exactly the same requested draw seeds. The baseline
is rerun and must reproduce the final c9 baseline artifact seed-for-seed,
including exclusions and optimiser-rescue telemetry, before any c20 artifact is
published.

The value 5.9 is an alternative sensitivity value. No rank-based estimate is
claimed because no independently reproducible rank-based estimator artifact is
part of this package.

Default invocation:
    python code/c20_nuc_sensitivity.py

Outputs:
    results/c20-nuc-sensitivity.csv
    results/c20-nuc-sensitivity-draws.npz
    results/c20-nuc-sensitivity-FINDING.md
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_relaxed_threshold_sensitivity as c2
import c7_ccc_garchx_bootstrap as c7
import c9_tcopula_bootstrap as c9

B = 2000
N_JOBS = 22
FIT_SEED = 12345
DRAW_SEED_OFFSET = 10_000
FIRST_DRAW_SEED = FIT_SEED + DRAW_SEED_OFFSET
LAST_DRAW_SEED = FIRST_DRAW_SEED + B - 1
RESCUE_SEED = 20260807
SEED_SCHEMA_VERSION = 2
ALT_NU_C = (5.9, 8.0)
C20_ANALYSIS_CONTRACT_VERSION = 1

C9_DRAWS_NAME = "c9-tcopula-draws-baseline.npz"
C9_RESULTS_NAME = "c9-tcopula-results.csv"
C20_CSV_NAME = "c20-nuc-sensitivity.csv"
C20_DRAWS_NAME = "c20-nuc-sensitivity-draws.npz"
C20_FINDING_NAME = "c20-nuc-sensitivity-FINDING.md"

TELEMETRY_KEYS = (
    "draw_had_six_start_rescue_attempt_t",
    "draw_retained_after_six_start_rescue_t",
    "n_asset_refits_with_six_start_rescue_attempt_t",
    "n_asset_refits_rescued_by_six_start_t",
)
HASH_KEYS = (
    "audit_design_sha256",
    "audit_fixed_null_dgp_sha256",
    "audit_refitter_sha256",
    "audit_simulator_sha256",
    "audit_analysis_stack_sha256",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_named_arrays(named_arrays):
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _scalar(payload, key):
    value = np.asarray(payload[key])
    require(value.shape == (), f"{key} must be scalar, got shape {value.shape}")
    return value.item()


def _text_scalar(payload, key):
    return str(_scalar(payload, key))


def _arrays_equal_exact(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _require_array_exact(name, actual, expected):
    require(_arrays_equal_exact(actual, expected), f"exact array mismatch: {name}")


def _require_float_exact(name, actual, expected):
    actual = float(actual)
    expected = float(expected)
    require(actual == expected, f"exact float mismatch for {name}: {actual!r} != {expected!r}")


def _validate_c9_reference(fixed):
    required = {
        "null_t",
        "null_t_full",
        "null_t_usable_mask",
        "null_t_excluded_mask",
        "draw_seeds",
        "seed_schema_version",
        "seed_schema",
        "base_seed",
        "draw_seed_offset",
        "first_draw_seed",
        "last_draw_seed",
        "B_requested",
        "d_bar_obs",
        "audit_spec",
        "audit_asset_names",
        "audit_nu_null_by_asset",
        "audit_nu_c_null",
        "audit_use_t_copula",
        *TELEMETRY_KEYS,
        *HASH_KEYS,
    }
    missing = sorted(required.difference(fixed))
    require(not missing, f"{C9_DRAWS_NAME} uses a stale schema: missing {missing}")

    require(int(_scalar(fixed, "seed_schema_version")) == SEED_SCHEMA_VERSION,
            "c9 baseline seed schema is not version 2")
    require(_text_scalar(fixed, "audit_spec") == "baseline",
            "c9 reference is not the baseline specification")
    require(bool(_scalar(fixed, "audit_use_t_copula")),
            "c9 baseline artifact does not use the Student-t copula")
    require(int(_scalar(fixed, "base_seed")) == FIT_SEED,
            "c9 baseline fit seed differs from 12345")
    require(int(_scalar(fixed, "draw_seed_offset")) == DRAW_SEED_OFFSET,
            "c9 baseline draw-seed offset differs from 10000")
    require(int(_scalar(fixed, "B_requested")) == B,
            "c9 baseline does not contain exactly 2000 requested draws")

    expected_seeds = np.arange(FIRST_DRAW_SEED, LAST_DRAW_SEED + 1, dtype=np.int64)
    draw_seeds = np.asarray(fixed["draw_seeds"], dtype=np.int64)
    _require_array_exact("c9 draw_seeds", draw_seeds, expected_seeds)
    require(
        (int(_scalar(fixed, "first_draw_seed")),
         int(_scalar(fixed, "last_draw_seed"))) == (FIRST_DRAW_SEED, LAST_DRAW_SEED),
        "c9 first/last draw-seed fields disagree with the required seed window",
    )

    full = np.asarray(fixed["null_t_full"], dtype=np.float64)
    usable = np.asarray(fixed["null_t_usable_mask"], dtype=np.bool_)
    excluded = np.asarray(fixed["null_t_excluded_mask"], dtype=np.bool_)
    compressed = np.asarray(fixed["null_t"], dtype=np.float64)
    require(full.shape == (B,), "c9 null_t_full must have shape (2000,)")
    require(usable.shape == (B,) and excluded.shape == (B,),
            "c9 usable/excluded masks must have shape (2000,)")
    _require_array_exact("c9 usable mask versus non-NaN positions", usable, ~np.isnan(full))
    _require_array_exact("c9 excluded mask", excluded, ~usable)
    _require_array_exact("c9 compressed null_t", compressed, full[usable])

    attempted = np.asarray(
        fixed["draw_had_six_start_rescue_attempt_t"], dtype=np.bool_
    )
    retained = np.asarray(
        fixed["draw_retained_after_six_start_rescue_t"], dtype=np.bool_
    )
    asset_attempts = np.asarray(
        fixed["n_asset_refits_with_six_start_rescue_attempt_t"], dtype=np.int64
    )
    asset_successes = np.asarray(
        fixed["n_asset_refits_rescued_by_six_start_t"], dtype=np.int64
    )
    for name, values in (
        ("rescue-attempt mask", attempted),
        ("rescue-retained mask", retained),
        ("asset rescue-attempt counts", asset_attempts),
        ("asset rescue-success counts", asset_successes),
    ):
        require(values.shape == (B,), f"c9 {name} must have shape (2000,)")
    _require_array_exact("c9 rescue-retained invariant", retained, attempted & usable)
    require(np.all(asset_attempts >= 0), "c9 asset rescue-attempt counts are negative")
    require(np.all(asset_successes >= 0), "c9 asset rescue-success counts are negative")
    require(np.all(asset_successes <= asset_attempts),
            "c9 asset rescue successes exceed attempts")
    _require_array_exact(
        "c9 asset rescue success total on non-rescue draws",
        asset_successes[~attempted],
        np.zeros(np.sum(~attempted), dtype=np.int64),
    )
    return expected_seeds


def _setup_current_baseline():
    design, _inf_dates, _reg_dates, ret_df = c7.build_design()
    observed = c7.fit_observed(design, seed=FIT_SEED)

    delta_infra = np.asarray(
        [observed[asset]["delta_infra"] for asset in c7.ASSETS], dtype=float
    )
    delta_reg = np.asarray(
        [observed[asset]["delta_reg"] for asset in c7.ASSETS], dtype=float
    )
    d_bar_obs = float(np.mean(delta_infra - delta_reg))
    multiplier = float(delta_infra.mean() / delta_reg.mean())

    R_return = ret_df.corr().values
    rho_return = float(c7.mean_off_diag(R_return))
    z_df = pd.DataFrame({
        asset: pd.Series(
            observed[asset]["z_resid"], index=design[asset]["index"]
        )
        for asset in c7.ASSETS
    }).dropna()
    R_z = z_df.corr().values
    rho_resid = float(c7.mean_off_diag(R_z))
    R_z_pd = R_z.copy()
    jitter = 0.0
    while True:
        try:
            L_z = np.linalg.cholesky(R_z_pd)
            break
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-8)
            R_z_pd = R_z + jitter * np.eye(len(c7.ASSETS))

    common_index = z_df.index
    common_pos = {
        asset: pd.Index(design[asset]["index"]).get_indexer(common_index)
        for asset in c7.ASSETS
    }

    c7._GLOBAL.clear()
    c7._GLOBAL.update({
        "design": design,
        "observed": observed,
        "R_z": R_z,
        "L_z": L_z,
        "n_common": len(common_index),
        "common_pos": common_pos,
        "max_len": max(
            observed[asset]["resid_unr"].shape[0] for asset in c7.ASSETS
        ),
        "rescue_seed": RESCUE_SEED,
    })
    c7.install_simulation_calendar(design)
    _nu_unrestricted, nu_null = c9._install_nu(observed)
    nu_null = np.asarray(nu_null, dtype=np.float64)
    nu_c_fitted = float(np.median(nu_null))
    audit = c9._fixed_dgp_audit_payload("baseline")
    return {
        "design": design,
        "observed": observed,
        "d_bar_obs": d_bar_obs,
        "multiplier": multiplier,
        "rho_return": rho_return,
        "rho_resid": rho_resid,
        "nu_null": nu_null,
        "nu_c_fitted": nu_c_fitted,
        "audit": audit,
    }


def _validate_local_baseline(local, fixed, c9_row):
    _require_float_exact("d_bar_obs local versus c9 NPZ",
                         local["d_bar_obs"], _scalar(fixed, "d_bar_obs"))
    _require_array_exact(
        "null-fit nu vector local versus c9",
        local["nu_null"],
        np.asarray(fixed["audit_nu_null_by_asset"], dtype=np.float64),
    )
    _require_float_exact(
        "fitted median nu_c local versus c9",
        local["nu_c_fitted"],
        _scalar(fixed, "audit_nu_c_null"),
    )
    _require_float_exact(
        "fitted median nu_c versus median null-fit nu vector",
        local["nu_c_fitted"],
        np.median(local["nu_null"]),
    )
    _require_array_exact(
        "asset ordering local versus c9",
        np.asarray(c7.ASSETS),
        np.asarray(fixed["audit_asset_names"]),
    )

    for key in HASH_KEYS:
        require(str(local["audit"][key]) == _text_scalar(fixed, key),
                f"locally reconstructed {key} differs from final c9 baseline")

    require(str(c9_row["spec"]) == "baseline", "c9 CSV baseline row is mislabeled")
    for name, actual, expected in (
        ("c9 CSV base_seed", c9_row["base_seed"], FIT_SEED),
        ("c9 CSV draw_seed_offset", c9_row["draw_seed_offset"], DRAW_SEED_OFFSET),
        ("c9 CSV first_draw_seed", c9_row["first_draw_seed"], FIRST_DRAW_SEED),
        ("c9 CSV last_draw_seed", c9_row["last_draw_seed"], LAST_DRAW_SEED),
        ("c9 CSV B_requested", c9_row["B_requested"], B),
    ):
        require(int(actual) == int(expected), f"{name} differs")
    _require_float_exact("c9 CSV d_bar_obs", c9_row["d_bar_obs"], local["d_bar_obs"])
    _require_float_exact(
        "c9 CSV fitted median nu_c",
        c9_row["nu_null_median"],
        local["nu_c_fitted"],
    )
    for name in ("multiplier", "rho_return", "rho_resid"):
        _require_float_exact(
            f"c9 CSV {name}",
            c9_row[name],
            local[name],
        )
    for csv_key, audit_key in (
        ("design_sha256", "audit_design_sha256"),
        ("fixed_null_dgp_sha256", "audit_fixed_null_dgp_sha256"),
        ("refitter_sha256", "audit_refitter_sha256"),
        ("simulator_sha256", "audit_simulator_sha256"),
        ("analysis_stack_sha256", "audit_analysis_stack_sha256"),
    ):
        require(str(c9_row[csv_key]) == _text_scalar(fixed, audit_key),
                f"c9 CSV {csv_key} differs from c9 NPZ")


def _summarize_draws(full, telemetry, d_bar_obs, baseline_full, baseline_usable):
    full = np.asarray(full, dtype=np.float64)
    require(full.shape == (B,), f"draw vector has shape {full.shape}, expected {(B,)}")
    usable = ~np.isnan(full)
    excluded = ~usable
    values = full[usable]
    require(len(values) > 0, "all c20 refits were excluded")

    attempted = np.asarray(
        telemetry["draw_had_six_start_rescue_attempt_t"], dtype=np.bool_
    )
    retained = np.asarray(
        telemetry["draw_retained_after_six_start_rescue_t"], dtype=np.bool_
    )
    asset_attempts = np.asarray(
        telemetry["n_asset_refits_with_six_start_rescue_attempt_t"], dtype=np.int64
    )
    asset_successes = np.asarray(
        telemetry["n_asset_refits_rescued_by_six_start_t"], dtype=np.int64
    )
    for key, array in (
        ("rescue attempted", attempted),
        ("rescue retained", retained),
        ("asset attempts", asset_attempts),
        ("asset successes", asset_successes),
    ):
        require(array.shape == (B,), f"{key} telemetry has shape {array.shape}")
    _require_array_exact("c20 rescue-retained invariant", retained, attempted & usable)
    require(np.all(asset_attempts >= 0), "c20 asset rescue-attempt counts are negative")
    require(np.all(asset_successes >= 0), "c20 asset rescue-success counts are negative")
    require(np.all(asset_successes <= asset_attempts),
            "c20 asset rescue successes exceed attempts")
    _require_array_exact(
        "c20 draw/asset rescue-attempt invariant",
        attempted,
        asset_attempts > 0,
    )

    upper_hits = int(np.sum(values >= d_bar_obs))
    abs_hits = int(np.sum(np.abs(values) >= abs(d_bar_obs)))
    lower_hits = int(np.sum(values <= d_bar_obs))
    B_used = int(usable.sum())
    p_one = (upper_hits + 1) / (B_used + 1)
    p_abs = (abs_hits + 1) / (B_used + 1)
    p_lower = (lower_hits + 1) / (B_used + 1)
    p_equal = min(1.0, 2.0 * min(p_one, p_lower))

    baseline_full = np.asarray(baseline_full, dtype=np.float64)
    baseline_usable = np.asarray(baseline_usable, dtype=np.bool_)
    paired_mask = usable & baseline_usable
    require(paired_mask.any(), "no seed positions are usable in both row and baseline")
    paired_difference = np.full(B, np.nan, dtype=np.float64)
    paired_difference[paired_mask] = full[paired_mask] - baseline_full[paired_mask]
    paired_values = paired_difference[paired_mask]

    summary = {
        "B_used": B_used,
        "n_dropped": int(excluded.sum()),
        "frac_dropped": float(excluded.mean()),
        "n_upper_tail_hits": upper_hits,
        "n_abs_stat_hits": abs_hits,
        "n_lower_tail_hits": lower_hits,
        "p_one": float(p_one),
        "p_abs": float(p_abs),
        "p_equal_tail": float(p_equal),
        "null_mean": float(values.mean()),
        "null_sd": float(values.std()),
        "n_draws_with_six_start_rescue_attempt": int(attempted.sum()),
        "n_draws_retained_after_six_start_rescue": int(retained.sum()),
        "n_draws_failed_after_six_start_rescue": int(np.sum(attempted & excluded)),
        "n_asset_refits_with_six_start_rescue_attempt": int(asset_attempts.sum()),
        "n_asset_refits_rescued_by_six_start": int(asset_successes.sum()),
        "n_paired_usable_with_baseline": int(paired_mask.sum()),
        "paired_diff_mean": float(paired_values.mean()),
        "paired_diff_sd": float(paired_values.std()),
        "paired_diff_max_abs": float(np.max(np.abs(paired_values))),
    }
    arrays = {
        "usable": usable,
        "excluded": excluded,
        "paired_mask": paired_mask,
        "paired_difference": paired_difference,
        "attempted": attempted,
        "retained": retained,
        "asset_attempts": asset_attempts,
        "asset_successes": asset_successes,
    }
    return summary, arrays


def _assert_baseline_reproduces_c9(
    full,
    arrays,
    summary,
    telemetry,
    fixed,
    c9_row,
):
    _require_array_exact(
        "c20 fitted-median full vector versus c9",
        np.asarray(full, dtype=np.float64),
        np.asarray(fixed["null_t_full"], dtype=np.float64),
    )
    _require_array_exact(
        "c20 fitted-median usable mask versus c9",
        arrays["usable"],
        np.asarray(fixed["null_t_usable_mask"], dtype=np.bool_),
    )
    _require_array_exact(
        "c20 fitted-median excluded mask versus c9",
        arrays["excluded"],
        np.asarray(fixed["null_t_excluded_mask"], dtype=np.bool_),
    )
    _require_array_exact(
        "c20 fitted-median compressed vector versus c9",
        np.asarray(full, dtype=np.float64)[arrays["usable"]],
        np.asarray(fixed["null_t"], dtype=np.float64),
    )
    for key in TELEMETRY_KEYS:
        _require_array_exact(
            f"c20 fitted-median {key} versus c9",
            telemetry[key],
            fixed[key],
        )

    canonical = np.asarray(fixed["null_t"], dtype=np.float64)
    _require_float_exact("baseline null mean", summary["null_mean"], canonical.mean())
    _require_float_exact("baseline null SD", summary["null_sd"], canonical.std())

    integer_matches = {
        "B_used_t": "B_used",
        "n_dropped_t": "n_dropped",
        "n_upper_tail_hits_t": "n_upper_tail_hits",
        "n_abs_stat_hits_t": "n_abs_stat_hits",
        "n_lower_tail_hits_t": "n_lower_tail_hits",
        "n_draws_with_six_start_rescue_attempt_t":
            "n_draws_with_six_start_rescue_attempt",
        "n_draws_retained_after_six_start_rescue_t":
            "n_draws_retained_after_six_start_rescue",
        "n_draws_failed_after_six_start_rescue_t":
            "n_draws_failed_after_six_start_rescue",
        "n_asset_refits_with_six_start_rescue_attempt_t":
            "n_asset_refits_with_six_start_rescue_attempt",
        "n_asset_refits_rescued_by_six_start_t":
            "n_asset_refits_rescued_by_six_start",
    }
    for csv_key, summary_key in integer_matches.items():
        require(int(c9_row[csv_key]) == int(summary[summary_key]),
                f"c20 baseline {summary_key} differs from c9 CSV {csv_key}")

    for name, actual, expected in (
        ("frac_dropped", summary["frac_dropped"], c9_row["frac_dropped_t"]),
        ("p_one", summary["p_one"], c9_row["p_tcopula_one_sided"]),
        ("p_abs", summary["p_abs"], c9_row["p_tcopula_two_sided"]),
        ("null_sd", summary["null_sd"], c9_row["null_sd_tcopula"]),
    ):
        _require_float_exact(f"c20 baseline {name} versus c9 CSV", actual, expected)


def _sensitivity_grid(fitted_nu_c):
    """Return the prespecified c20 roles, provenance labels, and copula-df grid."""
    roles = (
        "fitted_median_baseline",
        "alternative_sensitivity",
        "alternative_sensitivity",
    )
    provenance = (
        "median of the six null-fit marginal nu values",
        "alternative sensitivity value; not a rank-based estimate",
        "alternative lighter-tail sensitivity value",
    )
    nu_grid = (float(fitted_nu_c), *ALT_NU_C)
    return roles, provenance, nu_grid


def current_c20_analysis_provenance(roles, provenance, nu_grid,
                                    c9_analysis_stack_sha256):
    """Bind c20's paired summaries and exact sensitivity grid to current source."""
    roles_array = np.asarray(roles)
    provenance_array = np.asarray(provenance)
    nu_grid_array = np.asarray(nu_grid, dtype=np.float64)
    sensitivity_laws_sha256 = _hash_named_arrays((
        ("nu_c_roles", roles_array),
        ("nu_c_provenance", provenance_array),
        ("nu_c_grid", nu_grid_array),
    ))
    contract = (
        f"c20-analysis-contract-v{C20_ANALYSIS_CONTRACT_VERSION}\n"
        f"B={B}\nN_JOBS={N_JOBS}\nFIT_SEED={FIT_SEED}\n"
        f"DRAW_SEED_OFFSET={DRAW_SEED_OFFSET}\nRESCUE_SEED={RESCUE_SEED}\n"
        f"SEED_SCHEMA_VERSION={SEED_SCHEMA_VERSION}\n"
        f"ALT_NU_C={ALT_NU_C!r}\n"
        + "\n".join(inspect.getsource(function) for function in (
            _validate_c9_reference,
            _setup_current_baseline,
            _validate_local_baseline,
            _summarize_draws,
            _assert_baseline_reproduces_c9,
            _sensitivity_grid,
            _write_outputs,
            main,
        ))
    )
    analysis_sha256 = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    source_sha256 = _sha256_file(Path(__file__).resolve())
    parent = str(c9_analysis_stack_sha256)
    stack_sha256 = hashlib.sha256(
        (
            f"c20-downstream-analysis-stack-v{C20_ANALYSIS_CONTRACT_VERSION}\n"
            f"{parent}\n{analysis_sha256}\n{sensitivity_laws_sha256}\n{source_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "c20_analysis_contract_version": C20_ANALYSIS_CONTRACT_VERSION,
        "c20_analysis_contract": contract,
        "c20_analysis_sha256": analysis_sha256,
        "c20_sensitivity_laws_sha256": sensitivity_laws_sha256,
        "c20_source_sha256": source_sha256,
        "c20_parent_c9_analysis_stack_sha256": parent,
        "c20_downstream_analysis_stack_sha256": stack_sha256,
    }


def _write_finding(frame, source_npz_sha256, source_csv_sha256, elapsed):
    baseline = frame.loc[frame["nu_c_role"] == "fitted_median_baseline"].iloc[0]
    p_span = float(frame["p_one"].max() - frame["p_one"].min())
    mean_span = float(frame["null_mean"].max() - frame["null_mean"].min())

    lines = [
        "# C20 -- Paired copula-df sensitivity",
        "",
        (
            f"_B={B} requested per row; fit seed {FIT_SEED}; requested draw seeds "
            f"{FIRST_DRAW_SEED}-{LAST_DRAW_SEED}; runtime {elapsed/60:.1f} min._"
        ),
        "",
        "## Pairing and provenance",
        "",
        (
            "All three rows use the same observed fit and the same ordered draw-seed "
            "array. The fitted-median row was rerun through c9's full diagnostic "
            "runner and matched the final c9 baseline exactly: full statistic vector "
            "including NaNs, usable/excluded masks, compressed vector, rescue "
            "telemetry, observed statistic, fitted marginal degrees of freedom, "
            "tail-hit counts, p-values, null mean, null SD, and audit hashes."
        ),
        "",
        (
            "The value 5.9 is an alternative sensitivity value, not a rank-based "
            "estimate. No independently reproducible rank-based estimation artifact "
            "is included in this package. The value 8.0 is a second, lighter-tail "
            "copula-df sensitivity value. Per-asset fitted Student-t margins remain "
            "unchanged in every row; only the shared copula df changes."
        ),
        "",
        "## Results",
        "",
        (
            "| role | nu_c | used/requested | one-sided p | absolute-statistic p | "
            "null mean | null SD | paired usable | mean paired change vs baseline |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['nu_c_role']} | {row['nu_c']:.6g} | "
            f"{int(row['B_used'])}/{int(row['B_requested'])} | "
            f"{row['p_one']:.4f} | {row['p_abs']:.4f} | "
            f"{row['null_mean']:.4f} | {row['null_sd']:.4f} | "
            f"{int(row['n_paired_usable_with_baseline'])} | "
            f"{row['paired_diff_mean']:+.4f} |"
        )
    lines.extend([
        "",
        (
            f"Across these three tested values, the one-sided p-value range has width "
            f"{p_span:.4f} and the null-mean range has width {mean_span:.4f}. "
            "Because exclusion masks can differ by copula df, rowwise p-values use "
            "each row's retained draws, whereas paired-change summaries use only the "
            "intersection of that row's and the fitted baseline's usable seed positions."
        ),
        "",
        "## Interpretation",
        "",
        (
            "This is a local, common-random-number sensitivity analysis. It does not "
            "estimate the copula df, validate the fixed-path bootstrap under the "
            "unknown data-generating process, or address the larger difference between "
            "the fixed-path and floored-recursive resampling schemes."
        ),
        "",
        "## Audit anchors",
        "",
        f"- fitted median nu_c: {baseline['nu_c']:.17g}",
        f"- c9 baseline NPZ SHA-256: {source_npz_sha256}",
        f"- c9 results CSV SHA-256: {source_csv_sha256}",
        f"- design hash: {baseline['design_sha256']}",
        f"- fixed-null DGP hash: {baseline['fixed_null_dgp_sha256']}",
        f"- refitter hash: {baseline['refitter_sha256']}",
        f"- simulator hash: {baseline['simulator_sha256']}",
        f"- analysis-stack hash: {baseline['analysis_stack_sha256']}",
        f"- c20 analysis hash: {baseline['c20_analysis_sha256']}",
        f"- c20 sensitivity-laws hash: {baseline['c20_sensitivity_laws_sha256']}",
        f"- c20 source hash: {baseline['c20_source_sha256']}",
        (
            "- c20 downstream analysis-stack hash: "
            f"{baseline['c20_downstream_analysis_stack_sha256']}"
        ),
        "",
        "## Files",
        "",
        f"- {C20_CSV_NAME}",
        f"- {C20_DRAWS_NAME}",
        "- code/c20_nuc_sensitivity.py",
        "",
    ])
    path = c2.OUT_DIR / C20_FINDING_NAME
    tmp = path.with_name(path.stem + ".write-tmp.md")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)


def _write_outputs(
    frame,
    full_matrix,
    usable_matrix,
    excluded_matrix,
    attempted_matrix,
    retained_matrix,
    asset_attempts_matrix,
    asset_successes_matrix,
    paired_mask_matrix,
    paired_difference_matrix,
    local,
    row_audits,
    expected_seeds,
    source_npz_sha256,
    source_csv_sha256,
    source_c9_code_sha256,
    c20_provenance,
    elapsed,
):
    csv_path = c2.OUT_DIR / C20_CSV_NAME
    csv_tmp = csv_path.with_name(csv_path.stem + ".write-tmp.csv")
    frame.to_csv(csv_tmp, index=False, float_format="%.17g")
    trial = pd.read_csv(csv_tmp, float_precision="round_trip")
    require(trial["nu_c_role"].tolist() == frame["nu_c_role"].tolist(),
            "c20 CSV row order failed round-trip validation")
    require(trial["B_requested"].astype(int).tolist() == [B, B, B],
            "c20 CSV requested counts failed round-trip validation")
    for key in frame.columns:
        if pd.api.types.is_float_dtype(frame[key]):
            _require_array_exact(
                f"c20 CSV float round trip: {key}",
                trial[key].to_numpy(dtype=np.float64),
                frame[key].to_numpy(dtype=np.float64),
            )
    for key in (
        "c20_analysis_sha256",
        "c20_sensitivity_laws_sha256",
        "c20_source_sha256",
        "c20_parent_c9_analysis_stack_sha256",
        "c20_downstream_analysis_stack_sha256",
    ):
        require(
            set(trial[key].astype(str)) == {str(c20_provenance[key])},
            f"c20 CSV changed {key}",
        )
    require(
        set(trial["source_c9_baseline_npz_sha256"].astype(str))
        == {str(source_npz_sha256)},
        "c20 CSV changed the source c9 NPZ hash",
    )
    require(
        set(trial["source_c9_results_csv_sha256"].astype(str))
        == {str(source_csv_sha256)},
        "c20 CSV changed the source c9 CSV hash",
    )
    csv_tmp.replace(csv_path)

    npz_path = c2.OUT_DIR / C20_DRAWS_NAME
    npz_tmp = npz_path.with_name(npz_path.stem + ".write-tmp.npz")
    np.savez(
        npz_tmp,
        audit_schema_version=np.int64(1),
        seed_schema_version=np.int64(SEED_SCHEMA_VERSION),
        seed_schema=np.asarray(
            "draw_seeds[i] = fit_seed + draw_seed_offset + i; all three rows "
            "preserve the same ordered 2000 seed positions and use NaN for exclusions"
        ),
        spec=np.asarray("baseline"),
        nu_c_roles=frame["nu_c_role"].to_numpy(dtype=str),
        nu_c_provenance=frame["nu_c_provenance"].to_numpy(dtype=str),
        nu_c_grid=frame["nu_c"].to_numpy(dtype=np.float64),
        fitted_nu_null_by_asset=local["nu_null"],
        fitted_nu_c_median=np.float64(local["nu_c_fitted"]),
        fit_seed=np.int64(FIT_SEED),
        draw_seed_offset=np.int64(DRAW_SEED_OFFSET),
        first_draw_seed=np.int64(FIRST_DRAW_SEED),
        last_draw_seed=np.int64(LAST_DRAW_SEED),
        B_requested=np.int64(B),
        draw_seeds=expected_seeds,
        d_bar_obs=np.float64(local["d_bar_obs"]),
        multiplier=np.float64(local["multiplier"]),
        rho_return=np.float64(local["rho_return"]),
        rho_resid=np.float64(local["rho_resid"]),
        null_t_full=full_matrix,
        null_t_usable_mask=usable_matrix,
        null_t_excluded_mask=excluded_matrix,
        draw_had_six_start_rescue_attempt=attempted_matrix,
        draw_retained_after_six_start_rescue=retained_matrix,
        n_asset_refits_with_six_start_rescue_attempt=asset_attempts_matrix,
        n_asset_refits_rescued_by_six_start=asset_successes_matrix,
        paired_usable_with_baseline_mask=paired_mask_matrix,
        paired_difference_vs_baseline=paired_difference_matrix,
        B_used=frame["B_used"].to_numpy(dtype=np.int64),
        n_dropped=frame["n_dropped"].to_numpy(dtype=np.int64),
        n_upper_tail_hits=frame["n_upper_tail_hits"].to_numpy(dtype=np.int64),
        n_abs_stat_hits=frame["n_abs_stat_hits"].to_numpy(dtype=np.int64),
        n_lower_tail_hits=frame["n_lower_tail_hits"].to_numpy(dtype=np.int64),
        p_one=frame["p_one"].to_numpy(dtype=np.float64),
        p_abs=frame["p_abs"].to_numpy(dtype=np.float64),
        p_equal_tail=frame["p_equal_tail"].to_numpy(dtype=np.float64),
        null_mean=frame["null_mean"].to_numpy(dtype=np.float64),
        null_sd=frame["null_sd"].to_numpy(dtype=np.float64),
        n_paired_usable_with_baseline=frame[
            "n_paired_usable_with_baseline"
        ].to_numpy(dtype=np.int64),
        paired_diff_mean=frame["paired_diff_mean"].to_numpy(dtype=np.float64),
        paired_diff_sd=frame["paired_diff_sd"].to_numpy(dtype=np.float64),
        paired_diff_max_abs=frame["paired_diff_max_abs"].to_numpy(dtype=np.float64),
        audit_design_sha256=np.asarray([
            str(audit["audit_design_sha256"]) for audit in row_audits
        ]),
        audit_fixed_null_dgp_sha256=np.asarray([
            str(audit["audit_fixed_null_dgp_sha256"]) for audit in row_audits
        ]),
        audit_refitter_sha256=np.asarray([
            str(audit["audit_refitter_sha256"]) for audit in row_audits
        ]),
        audit_simulator_sha256=np.asarray([
            str(audit["audit_simulator_sha256"]) for audit in row_audits
        ]),
        audit_analysis_stack_sha256=np.asarray([
            str(audit["audit_analysis_stack_sha256"]) for audit in row_audits
        ]),
        source_c9_baseline_npz=np.asarray(C9_DRAWS_NAME),
        source_c9_baseline_npz_sha256=np.asarray(source_npz_sha256),
        source_c9_results_csv=np.asarray(C9_RESULTS_NAME),
        source_c9_results_csv_sha256=np.asarray(source_csv_sha256),
        source_c9_code_sha256=np.asarray(source_c9_code_sha256),
        baseline_exact_match_c9=np.bool_(True),
        c20_analysis_contract_version=np.int64(
            c20_provenance["c20_analysis_contract_version"]
        ),
        c20_analysis_contract=np.asarray(
            c20_provenance["c20_analysis_contract"]
        ),
        c20_analysis_sha256=np.asarray(c20_provenance["c20_analysis_sha256"]),
        c20_sensitivity_laws_sha256=np.asarray(
            c20_provenance["c20_sensitivity_laws_sha256"]
        ),
        c20_source_sha256=np.asarray(c20_provenance["c20_source_sha256"]),
        c20_parent_c9_analysis_stack_sha256=np.asarray(
            c20_provenance["c20_parent_c9_analysis_stack_sha256"]
        ),
        c20_downstream_analysis_stack_sha256=np.asarray(
            c20_provenance["c20_downstream_analysis_stack_sha256"]
        ),
    )
    with np.load(npz_tmp, allow_pickle=False) as trial_npz:
        require(trial_npz["null_t_full"].shape == (3, B),
                "c20 NPZ full draw matrix failed round-trip validation")
        _require_array_exact(
            "c20 NPZ draw seeds round trip",
            trial_npz["draw_seeds"],
            expected_seeds,
        )
        require(bool(trial_npz["baseline_exact_match_c9"]),
                "c20 NPZ lost the exact c9 baseline-match flag")
        require(
            _text_scalar(trial_npz, "source_c9_baseline_npz_sha256")
            == str(source_npz_sha256),
            "c20 NPZ changed the source c9 NPZ hash",
        )
        require(
            _text_scalar(trial_npz, "source_c9_results_csv_sha256")
            == str(source_csv_sha256),
            "c20 NPZ changed the source c9 CSV hash",
        )
        trial_laws_sha256 = _hash_named_arrays((
            ("nu_c_roles", trial_npz["nu_c_roles"]),
            ("nu_c_provenance", trial_npz["nu_c_provenance"]),
            ("nu_c_grid", trial_npz["nu_c_grid"]),
        ))
        require(
            trial_laws_sha256 == c20_provenance["c20_sensitivity_laws_sha256"],
            "c20 NPZ changed the sensitivity grid",
        )
        for key in (
            "c20_analysis_contract",
            "c20_analysis_sha256",
            "c20_sensitivity_laws_sha256",
            "c20_source_sha256",
            "c20_parent_c9_analysis_stack_sha256",
            "c20_downstream_analysis_stack_sha256",
        ):
            require(
                _text_scalar(trial_npz, key) == str(c20_provenance[key]),
                f"c20 NPZ changed {key}",
            )
    npz_tmp.replace(npz_path)
    _write_finding(frame, source_npz_sha256, source_csv_sha256, elapsed)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed B=2000 paired copula-df sensitivity against the "
            "committed c9 baseline."
        )
    )
    parser.parse_args()
    start = time.time()
    require(c9.USE_T_COPULA, "c20 requires c9.USE_T_COPULA=True")

    c9_npz_path = c2.OUT_DIR / C9_DRAWS_NAME
    c9_csv_path = c2.OUT_DIR / C9_RESULTS_NAME
    require(c9_npz_path.is_file(), f"missing final c9 baseline artifact: {c9_npz_path}")
    require(c9_csv_path.is_file(), f"missing final c9 results artifact: {c9_csv_path}")

    fixed = _load_npz(c9_npz_path)
    expected_seeds = _validate_c9_reference(fixed)
    # Preserve IEEE-754 values written with enough decimal digits for an exact
    # round trip.  Pandas' default fast parser can choose the adjacent float.
    c9_results = pd.read_csv(c9_csv_path, float_precision="round_trip")
    baseline_rows = c9_results.loc[c9_results["spec"] == "baseline"]
    require(len(baseline_rows) == 1, "c9 results must contain exactly one baseline row")
    c9_row = baseline_rows.iloc[0]

    print("Reconstructing the c9 baseline fit and audit fingerprints...")
    local = _setup_current_baseline()
    _validate_local_baseline(local, fixed, c9_row)
    print(
        f"  d_bar_obs={local['d_bar_obs']:.6f}; "
        f"fitted median nu_c={local['nu_c_fitted']:.6f}"
    )
    print(
        f"  seeds={FIRST_DRAW_SEED}-{LAST_DRAW_SEED}; "
        f"analysis={str(local['audit']['audit_analysis_stack_sha256'])[:16]}..."
    )

    roles, provenance, nu_grid = _sensitivity_grid(local["nu_c_fitted"])
    c20_provenance = current_c20_analysis_provenance(
        roles,
        provenance,
        nu_grid,
        _text_scalar(fixed, "audit_analysis_stack_sha256"),
    )

    rows = []
    full_rows = []
    usable_rows = []
    excluded_rows = []
    attempted_rows = []
    retained_rows = []
    asset_attempt_rows = []
    asset_success_rows = []
    paired_mask_rows = []
    paired_difference_rows = []
    row_audits = []

    canonical_full = np.asarray(fixed["null_t_full"], dtype=np.float64)
    canonical_usable = np.asarray(fixed["null_t_usable_mask"], dtype=np.bool_)
    source_npz_sha256 = _sha256_file(c9_npz_path)
    source_csv_sha256 = _sha256_file(c9_csv_path)
    source_c9_code_sha256 = _sha256_file(HERE / "c9_tcopula_bootstrap.py")

    for row_index, (role, source, nu_c) in enumerate(
        zip(roles, provenance, nu_grid)
    ):
        c7._GLOBAL["nu_c_null"] = float(nu_c)
        row_audit = c9._fixed_dgp_audit_payload("baseline")
        print(
            f"[{row_index + 1}/3] role={role}; nu_c={nu_c:.6f}; "
            f"B={B}; seeds={FIRST_DRAW_SEED}-{LAST_DRAW_SEED}"
        )
        draw_seeds, full, telemetry = c9._run_t_bootstrap_full(
            B, N_JOBS, FIRST_DRAW_SEED
        )
        _require_array_exact(f"{role} draw seeds", draw_seeds, expected_seeds)

        summary, arrays = _summarize_draws(
            full,
            telemetry,
            local["d_bar_obs"],
            canonical_full,
            canonical_usable,
        )
        if row_index == 0:
            _assert_baseline_reproduces_c9(
                full, arrays, summary, telemetry, fixed, c9_row
            )
            require(
                str(row_audit["audit_analysis_stack_sha256"])
                == _text_scalar(fixed, "audit_analysis_stack_sha256"),
                "rerun fitted-median analysis hash differs from final c9",
            )
            print("  exact seed-for-seed c9 baseline reproduction: PASS")

        row = {
            "spec": "baseline",
            "nu_c_role": role,
            "nu_c": float(nu_c),
            "nu_c_provenance": source,
            "fit_seed": FIT_SEED,
            "draw_seed_offset": DRAW_SEED_OFFSET,
            "first_draw_seed": FIRST_DRAW_SEED,
            "last_draw_seed": LAST_DRAW_SEED,
            "B_requested": B,
            **summary,
            "d_bar_obs": local["d_bar_obs"],
            "multiplier": local["multiplier"],
            "rho_return": local["rho_return"],
            "rho_resid": local["rho_resid"],
            "design_sha256": str(row_audit["audit_design_sha256"]),
            "fixed_null_dgp_sha256": str(
                row_audit["audit_fixed_null_dgp_sha256"]
            ),
            "refitter_sha256": str(row_audit["audit_refitter_sha256"]),
            "simulator_sha256": str(row_audit["audit_simulator_sha256"]),
            "analysis_stack_sha256": str(
                row_audit["audit_analysis_stack_sha256"]
            ),
            "source_c9_baseline_npz_sha256": source_npz_sha256,
            "source_c9_results_csv_sha256": source_csv_sha256,
            "c9_baseline_comparison":
                "exact_match" if row_index == 0 else "not_applicable",
            "c20_analysis_contract_version": c20_provenance[
                "c20_analysis_contract_version"
            ],
            "c20_analysis_sha256": c20_provenance["c20_analysis_sha256"],
            "c20_sensitivity_laws_sha256": c20_provenance[
                "c20_sensitivity_laws_sha256"
            ],
            "c20_source_sha256": c20_provenance["c20_source_sha256"],
            "c20_parent_c9_analysis_stack_sha256": c20_provenance[
                "c20_parent_c9_analysis_stack_sha256"
            ],
            "c20_downstream_analysis_stack_sha256": c20_provenance[
                "c20_downstream_analysis_stack_sha256"
            ],
        }
        rows.append(row)
        full_rows.append(np.asarray(full, dtype=np.float64))
        usable_rows.append(arrays["usable"])
        excluded_rows.append(arrays["excluded"])
        attempted_rows.append(arrays["attempted"])
        retained_rows.append(arrays["retained"])
        asset_attempt_rows.append(arrays["asset_attempts"])
        asset_success_rows.append(arrays["asset_successes"])
        paired_mask_rows.append(arrays["paired_mask"])
        paired_difference_rows.append(arrays["paired_difference"])
        row_audits.append(row_audit)

        print(
            f"  used={summary['B_used']}; dropped={summary['n_dropped']}; "
            f"p_one={summary['p_one']:.4f}; p_abs={summary['p_abs']:.4f}; "
            f"mean={summary['null_mean']:.4f}; SD={summary['null_sd']:.4f}"
        )

    c7._GLOBAL["nu_c_null"] = local["nu_c_fitted"]
    frame = pd.DataFrame(rows)
    if _sha256_file(c9_npz_path) != source_npz_sha256:
        raise RuntimeError("source c9 baseline archive changed during c20 computation")
    if _sha256_file(c9_csv_path) != source_csv_sha256:
        raise RuntimeError("source c9 results CSV changed during c20 computation")
    if _sha256_file(HERE / "c9_tcopula_bootstrap.py") != source_c9_code_sha256:
        raise RuntimeError("source c9 code changed during c20 computation")
    if current_c20_analysis_provenance(
        roles,
        provenance,
        nu_grid,
        _text_scalar(fixed, "audit_analysis_stack_sha256"),
    ) != c20_provenance:
        raise RuntimeError(
            "c20 source/analysis contract changed during computation; rerun cleanly"
        )
    elapsed = time.time() - start
    _write_outputs(
        frame=frame,
        full_matrix=np.stack(full_rows),
        usable_matrix=np.stack(usable_rows),
        excluded_matrix=np.stack(excluded_rows),
        attempted_matrix=np.stack(attempted_rows),
        retained_matrix=np.stack(retained_rows),
        asset_attempts_matrix=np.stack(asset_attempt_rows),
        asset_successes_matrix=np.stack(asset_success_rows),
        paired_mask_matrix=np.stack(paired_mask_rows),
        paired_difference_matrix=np.stack(paired_difference_rows),
        local=local,
        row_audits=row_audits,
        expected_seeds=expected_seeds,
        source_npz_sha256=source_npz_sha256,
        source_csv_sha256=source_csv_sha256,
        source_c9_code_sha256=source_c9_code_sha256,
        c20_provenance=c20_provenance,
        elapsed=elapsed,
    )
    print(f"Saved {c2.OUT_DIR / C20_CSV_NAME}")
    print(f"Saved {c2.OUT_DIR / C20_DRAWS_NAME}")
    print(f"Saved {c2.OUT_DIR / C20_FINDING_NAME}")
    print(f"Total {elapsed:.1f}s")
    print("C20_NUC_DONE")


if __name__ == "__main__":
    main()
