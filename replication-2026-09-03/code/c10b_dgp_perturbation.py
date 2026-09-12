"""
C10b: DGP perturbations for the c10 internal size-calibration study.

All scenarios use deliberate common random numbers (the same seed-indexed outer
panels and reference streams).  S0 is not an independent replication: it is an
exact lower-budget prefix of the final c10 archive (N=300, B_ref=2000 by
default), asserted before any perturbation computation.  The Gaussian reference
law is invariant to the Student-t margin/copula-df perturbations, so its verified
c10 prefix is cached once.  Student-t reference draws are cached by calibration
(nu vector, nu_c), which also avoids repeating S4's fitted-nu calibration.

Run:
    python c10b_dgp_perturbation.py --N 300 --B-ref 2000 --n-jobs 22 \
        --fit-seed 12345 --mc-seed 20260618

Outputs:
    results/c10b-dgp-perturbation-results.csv
    results/c10b-dgp-perturbation-draws.npz
    results/c10b-dgp-perturbation-FINDING.md
"""
import argparse
import hashlib
import inspect
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import c7_ccc_garchx_bootstrap as c7
import c10_size_study as c10

ASSETS = c10.ASSETS
_G = c10._G
C10B_ANALYSIS_CONTRACT_VERSION = 1
C10B_SEED_SCHEMA_VERSION = 1


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_named_arrays(named_arrays):
    """Hash names, dtypes, shapes, and bytes of scenario-law arrays."""
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


def _scenario_law_sha256(definition):
    """Hash one row's exact simulation law, calibration law, and S0 role."""
    return _hash_named_arrays((
        ("scenario_name", np.asarray(str(definition["name"]))),
        (
            "scenario_is_s0_prefix",
            np.asarray(bool(definition["is_s0_prefix"]), dtype=np.bool_),
        ),
        ("scenario_nu_sim", np.asarray(definition["nu_sim"], dtype=np.float64)),
        ("scenario_nu_cal", np.asarray(definition["nu_cal"], dtype=np.float64)),
        (
            "scenario_nu_c_sim",
            np.asarray(float(definition["nu_c_sim"]), dtype=np.float64),
        ),
        (
            "scenario_nu_c_cal",
            np.asarray(float(definition["nu_c_cal"]), dtype=np.float64),
        ),
    ))


def setup(fit_seed=c10.FIT_SEED_DEFAULT):
    """Install and verify the exact c9 baseline DGP through c10's shared gate."""
    return c10.setup_fitted_dgp(fit_seed=fit_seed, require_c9=True)


def _require_c10_prefix(setup_result, fit_seed, mc_seed, N, B_ref):
    """Load and validate the exact seed-indexed S0 prefix from final c10."""
    path = c10.c2.OUT_DIR / "c10-size-study-draws.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "c10b requires the final c10 draw archive before computation: "
            f"{path}"
        )
    required = {
        "c10_seed_schema_version", "c9_seed_schema_version",
        "fit_seed", "mc_seed", "rescue_seed", "N_requested", "B_ref_requested",
        "asset_names", "panel_seeds", "panel_usable_mask", "panel_excluded_mask",
        "panel_se_ok_full", "panel_delta_infra_full", "panel_delta_reg_full",
        "panel_se_infra_full", "panel_se_reg_full",
        "panel_dbar_full", "panel_p_iid_full", "panel_p_de_full",
        "gaussian_reference_seeds", "gaussian_reference_full",
        "gaussian_reference_usable_mask", "gaussian_reference_excluded_mask",
        "t_reference_seeds", "t_reference_full", "t_reference_usable_mask",
        "t_reference_excluded_mask", "dbar_panels", "p_iid", "p_de",
        "g_draws", "t_draws", "audit_design_sha256",
        "audit_fixed_null_dgp_sha256", "audit_refitter_sha256",
        "audit_simulator_sha256", "audit_analysis_stack_sha256",
        "c9_exact_identity_verified", "null_params_by_asset",
        "null_mean_returns", "null_sigma2_concat", "null_sigma2_offsets",
        "nu_null_by_asset", "nu_c_null", "R_z", "L_z",
        "simulation_calendar_ns", "calendar_pos_concat", "calendar_pos_offsets",
        "availability_matrix", "availability_pattern_code",
        "availability_pattern_codes", "availability_pattern_counts",
        "availability_pattern_first_date_ns", "availability_pattern_last_date_ns",
        "c10_analysis_contract_version", "c10_analysis_contract",
        "c10_analysis_sha256", "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(
                f"{path.name} is not a final provenance-complete c10 archive: "
                f"{sorted(missing)}"
            )
        fixed = {key: archive[key] for key in archive.files}

    if int(fixed["c10_seed_schema_version"]) != c10.C10_SEED_SCHEMA_VERSION:
        raise RuntimeError("unsupported c10 seed schema")
    if int(fixed["c9_seed_schema_version"]) != 2:
        raise RuntimeError("c10 prefix was not certified against c9 schema v2")
    if not bool(fixed["c9_exact_identity_verified"]):
        raise RuntimeError("c10 archive does not certify exact c9 identity")
    if int(fixed["fit_seed"]) != int(fit_seed):
        raise RuntimeError("c10b fit seed differs from final c10")
    if int(fixed["mc_seed"]) != int(mc_seed):
        raise RuntimeError("c10b mc seed differs from final c10 S0 stream")
    if int(fixed["rescue_seed"]) != c10.RESCUE_SEED:
        raise RuntimeError("c10 rescue seed differs from c10b contract")
    if int(fixed["N_requested"]) < N or int(fixed["B_ref_requested"]) < B_ref:
        raise RuntimeError(
            "c10 archive is too short for the requested c10b S0 prefix: "
            f"need N={N}, B_ref={B_ref}"
        )
    if not np.array_equal(fixed["asset_names"], np.asarray(ASSETS)):
        raise RuntimeError("c10 asset order differs from c10b")

    current_c10 = c10.current_c10_analysis_provenance(
        setup_result["provenance"]["audit_analysis_stack_sha256"]
    )
    if int(fixed["c10_analysis_contract_version"]) != int(
        current_c10["c10_analysis_contract_version"]
    ):
        raise RuntimeError("c10 archive analysis-contract version is stale")
    for key in (
        "c10_analysis_contract",
        "c10_analysis_sha256",
        "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    ):
        if c10._scalar_text(fixed[key]) != str(current_c10[key]):
            raise RuntimeError(f"c10 archive {key} differs from current c10 source")

    audit = setup_result["audit_payload"]
    for key in c10._C9_HASH_KEYS:
        if c10._scalar_text(fixed[key]) != c10._scalar_text(audit[key]):
            raise RuntimeError(f"c10 {key} differs from local/c9 baseline")
    dgp_fields = {
        "asset_names": "audit_asset_names",
        "null_params_by_asset": "audit_null_params_by_asset",
        "null_mean_returns": "audit_null_mean_returns",
        "null_sigma2_concat": "audit_null_sigma2_concat",
        "null_sigma2_offsets": "audit_null_sigma2_offsets",
        "nu_null_by_asset": "audit_nu_null_by_asset",
        "nu_c_null": "audit_nu_c_null",
        "R_z": "audit_R_z",
        "L_z": "audit_L_z",
        "simulation_calendar_ns": "audit_simulation_calendar_ns",
        "calendar_pos_concat": "audit_calendar_pos_concat",
        "calendar_pos_offsets": "audit_calendar_pos_offsets",
        "availability_matrix": "audit_availability_matrix",
        "availability_pattern_code": "audit_availability_pattern_code",
        "availability_pattern_codes": "audit_availability_pattern_codes",
        "availability_pattern_counts": "audit_availability_pattern_counts",
        "availability_pattern_first_date_ns": (
            "audit_availability_pattern_first_date_ns"
        ),
        "availability_pattern_last_date_ns": (
            "audit_availability_pattern_last_date_ns"
        ),
    }
    for c10_key, audit_key in dgp_fields.items():
        if not np.array_equal(fixed[c10_key], audit[audit_key]):
            raise RuntimeError(
                f"c10 {c10_key} differs exactly from local/c9 baseline"
            )

    expected_panel_seeds = np.arange(mc_seed, mc_seed + N, dtype=np.int64)
    expected_g_seeds = np.arange(
        mc_seed + c10.REFERENCE_SEED_OFFSET,
        mc_seed + c10.REFERENCE_SEED_OFFSET + B_ref,
        dtype=np.int64,
    )
    expected_t_seeds = expected_g_seeds + c10.T_REFERENCE_SEED_OFFSET
    if not np.array_equal(fixed["panel_seeds"][:N], expected_panel_seeds):
        raise RuntimeError("c10 panel seed prefix is not the c10b CRN stream")
    if not np.array_equal(fixed["gaussian_reference_seeds"][:B_ref], expected_g_seeds):
        raise RuntimeError("c10 Gaussian reference seed prefix differs")
    if not np.array_equal(fixed["t_reference_seeds"][:B_ref], expected_t_seeds):
        raise RuntimeError("c10 Student-t reference seed prefix differs")

    panel_mask = np.asarray(fixed["panel_usable_mask"][:N], dtype=bool)
    panel_excluded = np.asarray(fixed["panel_excluded_mask"][:N], dtype=bool)
    g_mask = np.asarray(fixed["gaussian_reference_usable_mask"][:B_ref], dtype=bool)
    t_mask = np.asarray(fixed["t_reference_usable_mask"][:B_ref], dtype=bool)
    if not np.array_equal(panel_excluded, ~panel_mask):
        raise RuntimeError("c10 panel masks are not complements")
    if not np.array_equal(
        fixed["gaussian_reference_excluded_mask"][:B_ref], ~g_mask
    ) or not np.array_equal(fixed["t_reference_excluded_mask"][:B_ref], ~t_mask):
        raise RuntimeError("c10 reference masks are not complements")
    if not np.array_equal(
        panel_mask, ~np.isnan(np.asarray(fixed["panel_dbar_full"][:N], dtype=float))
    ):
        raise RuntimeError("c10 panel mask does not match seed-indexed d_bar NaNs")
    if not np.array_equal(
        g_mask, ~np.isnan(np.asarray(fixed["gaussian_reference_full"][:B_ref], dtype=float))
    ) or not np.array_equal(
        t_mask, ~np.isnan(np.asarray(fixed["t_reference_full"][:B_ref], dtype=float))
    ):
        raise RuntimeError("c10 reference masks do not match seed-indexed NaNs")
    if not np.array_equal(
        fixed["dbar_panels"][:int(panel_mask.sum())],
        fixed["panel_dbar_full"][:N][panel_mask],
    ):
        raise RuntimeError("c10 legacy and full panel arrays disagree on S0 prefix")
    if not np.array_equal(fixed["g_draws"][:int(g_mask.sum())],
                          fixed["gaussian_reference_full"][:B_ref][g_mask]):
        raise RuntimeError("c10 legacy and full Gaussian arrays disagree")
    if not np.array_equal(fixed["t_draws"][:int(t_mask.sum())],
                          fixed["t_reference_full"][:B_ref][t_mask]):
        raise RuntimeError("c10 legacy and full Student-t arrays disagree")
    if not np.array_equal(
        fixed["p_iid"][:int(panel_mask.sum())],
        fixed["panel_p_iid_full"][:N][panel_mask],
        equal_nan=True,
    ) or not np.array_equal(
        fixed["p_de"][:int(panel_mask.sum())],
        fixed["panel_p_de_full"][:N][panel_mask],
        equal_nan=True,
    ):
        raise RuntimeError("c10 legacy and full panel p-value arrays disagree")

    return {
        # Store a package-relative provenance label rather than leaking the
        # builder's absolute checkout path into the portable result archive.
        "path": f"results/{path.name}",
        "archive_sha256": _sha256_file(path),
        "c10_provenance": current_c10,
        "panel_seeds": expected_panel_seeds,
        "panel_usable_mask": panel_mask,
        "panel_se_ok_full": np.asarray(fixed["panel_se_ok_full"][:N], dtype=bool),
        "delta_infra_full": np.asarray(fixed["panel_delta_infra_full"][:N], dtype=float),
        "delta_reg_full": np.asarray(fixed["panel_delta_reg_full"][:N], dtype=float),
        "se_infra_full": np.asarray(fixed["panel_se_infra_full"][:N], dtype=float),
        "se_reg_full": np.asarray(fixed["panel_se_reg_full"][:N], dtype=float),
        "dbar_full": np.asarray(fixed["panel_dbar_full"][:N], dtype=float),
        "p_iid_full": np.asarray(fixed["panel_p_iid_full"][:N], dtype=float),
        "p_de_full": np.asarray(fixed["panel_p_de_full"][:N], dtype=float),
        "gaussian": {
            "seeds": expected_g_seeds,
            "full": np.asarray(fixed["gaussian_reference_full"][:B_ref], dtype=float),
            "usable_mask": g_mask,
        },
        "student_t": {
            "seeds": expected_t_seeds,
            "full": np.asarray(fixed["t_reference_full"][:B_ref], dtype=float),
            "usable_mask": t_mask,
        },
    }


def _panel_results_from_prefix(prefix):
    """Reconstruct c10's accepted panel tuples for exact S0 re-evaluation."""
    results = []
    for i, usable in enumerate(prefix["panel_usable_mask"]):
        if not usable:
            results.append(None)
            continue
        results.append((
            prefix["delta_infra_full"][i],
            prefix["delta_reg_full"][i],
            prefix["se_infra_full"][i],
            prefix["se_reg_full"][i],
            float(prefix["dbar_full"][i]),
            bool(prefix["panel_se_ok_full"][i]),
        ))
    return results


def _reference_key(nu, nu_c):
    nu = np.ascontiguousarray(np.asarray(nu, dtype="<f8"))
    return (nu.tobytes(), np.float64(nu_c).tobytes())


def _run_panel_stream(N, n_jobs, panel_seeds):
    with Pool(processes=n_jobs) as pool:
        return pool.map(
            c10._panel_estimate,
            panel_seeds.tolist(),
            chunksize=max(1, N // (n_jobs * 4)),
        )


def run_scenario(definition, N, B_ref, n_jobs, rho_return, panel_seeds,
                 gaussian_reference, t_reference_cache, s0_prefix=None):
    """Run one CRN scenario and return its CSV row plus full audit payload."""
    name = definition["name"]
    nu_sim = np.asarray(definition["nu_sim"], dtype=float)
    nu_cal = np.asarray(definition["nu_cal"], dtype=float)
    nu_c_sim = float(definition["nu_c_sim"])
    nu_c_cal = float(definition["nu_c_cal"])

    key = _reference_key(nu_cal, nu_c_cal)
    if key not in t_reference_cache:
        _G["nu_null"] = nu_cal
        _G["nu_c_null"] = nu_c_cal
        c7._GLOBAL.update(_G)
        t_reference_cache[key] = c10.run_reference_distribution_full(
            c10._draw_dbar_null_tcopula,
            B_ref,
            n_jobs,
            int(gaussian_reference["seeds"][0]) + c10.T_REFERENCE_SEED_OFFSET,
        )
    t_reference = t_reference_cache[key]
    g_values = gaussian_reference["full"][gaussian_reference["usable_mask"]]
    t_values = t_reference["full"][t_reference["usable_mask"]]
    crit = {
        "gauss": c10.crit_from_draws(g_values),
        "t": c10.crit_from_draws(t_values),
    }

    _G["nu_null"] = nu_sim
    _G["nu_c_null"] = nu_c_sim
    c7._GLOBAL.update(_G)
    panel_results = (
        _panel_results_from_prefix(s0_prefix)
        if s0_prefix is not None
        else _run_panel_stream(N, n_jobs, panel_seeds)
    )
    panel = c10.evaluate_panel_results(
        panel_results, panel_seeds, rho_return, crit
    )
    if s0_prefix is not None:
        for key_name in (
            "panel_usable_mask", "delta_infra_full", "delta_reg_full",
            "se_infra_full", "se_reg_full", "dbar_full", "p_iid_full",
            "p_de_full",
        ):
            if not np.array_equal(panel[key_name], s0_prefix[key_name], equal_nan=True):
                raise RuntimeError(f"S0 c10 prefix assertion failed for {key_name}")

    n = int(panel["panel_usable_mask"].sum())
    methods = ["naive_iid", "design_effect", "gaussian_copula_boot", "tcopula_boot"]
    row = {
        "scenario": name,
        "N_used": n,
        "n_calendar": int(_G["n_calendar"]),
        "availability_pattern_codes": ";".join(
            str(int(v)) for v in _G["availability_pattern_codes"]
        ),
        "availability_pattern_counts": ";".join(
            str(int(v)) for v in _G["availability_pattern_counts"]
        ),
    }
    print(f"\n[{name}]  N_used={n}")
    for method in methods:
        size5 = panel["counts"][method][0.05] / n
        size10 = panel["counts"][method][0.10] / n
        se5 = np.sqrt(size5 * (1.0 - size5) / n)
        print(
            f"   {method:<22} size@5%={size5:5.3f}+/-{se5:.3f}   "
            f"size@10%={size10:5.3f}"
        )
        row[f"{method}_size05"] = size5
        row[f"{method}_se05"] = se5
        row[f"{method}_size10"] = size10
    return {
        "row": row,
        "panel": panel,
        "gaussian_reference": gaussian_reference,
        "t_reference": t_reference,
        "crit": crit,
        **definition,
    }


def _finding_text(rows, N, B_ref, fit_seed, mc_seed):
    """Render the result digest from the same rows committed to the CSV."""
    lines = [
        "# C10b -- DGP-perturbation sensitivity",
        "",
        (f"_N={N} common-random-number panel positions per scenario; "
         f"B_ref={B_ref}; fit seed={fit_seed}; Monte Carlo seed={mc_seed}._"),
        "",
        "This is a sensitivity study within specified fitted null DGPs, not "
        "independent validation under the unknown empirical DGP. S0 is asserted "
        "to be the exact seed-indexed prefix of the final c10 run; every scenario "
        "uses the same panel and reference seed positions.",
        "",
        "| scenario | usable | naive iid | design-effect t(5) | Gaussian comparator | Student-t copula |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {int(row['N_used'])} | "
            f"{row['naive_iid_size05']:.3f} | "
            f"{row['design_effect_size05']:.3f} | "
            f"{row['gaussian_copula_boot_size05']:.3f} | "
            f"{row['tcopula_boot_size05']:.3f} |"
        )
    lines.extend([
        "",
        "Entries are empirical rejection rates at nominal 5%; the CSV also "
        "contains the 10% rates and binomial Monte Carlo standard errors. The "
        "near-Gaussian row retains fitted Student-t margins and sets the "
        "Student-t copula df to 200; it is not a fully Gaussian innovation law. "
        "The deliberately misspecified final row separates simulation and "
        "calibration tails.",
        "",
        "Files: `c10b-dgp-perturbation-results.csv` and "
        "`c10b-dgp-perturbation-draws.npz`.",
        "",
    ])
    return "\n".join(lines)


def _scenario_definitions(nu_fit):
    """Return the five prespecified c10b simulation/calibration laws."""
    nu_fit = np.asarray(nu_fit, dtype=float)
    nu8 = np.full(len(ASSETS), 8.0)
    nu25 = np.full(len(ASSETS), 2.5)
    return [
        {
            "name": "S0 baseline (fitted nu, matched)",
            "nu_sim": nu_fit, "nu_c_sim": float(np.median(nu_fit)),
            "nu_cal": nu_fit, "nu_c_cal": float(np.median(nu_fit)),
            "is_s0_prefix": True,
        },
        {
            "name": "S1 lighter tails nu=8 (matched)",
            "nu_sim": nu8, "nu_c_sim": 8.0,
            "nu_cal": nu8, "nu_c_cal": 8.0,
            "is_s0_prefix": False,
        },
        {
            "name": "S2 heavier tails nu=2.5 (matched)",
            "nu_sim": nu25, "nu_c_sim": 2.5,
            "nu_cal": nu25, "nu_c_cal": 2.5,
            "is_s0_prefix": False,
        },
        {
            "name": "S3 near-Gaussian t-copula (nu_c=200, fitted margins)",
            "nu_sim": nu_fit, "nu_c_sim": 200.0,
            "nu_cal": nu_fit, "nu_c_cal": 200.0,
            "is_s0_prefix": False,
        },
        {
            "name": "S4 MIS-SPECIFIED (simulate nu=2.5, calibrate at fitted nu)",
            "nu_sim": nu25, "nu_c_sim": 2.5,
            "nu_cal": nu_fit, "nu_c_cal": float(np.median(nu_fit)),
            "is_s0_prefix": False,
        },
    ]


def current_c10b_analysis_provenance(scenarios, c10_parent_stack_sha256):
    """Bind c10b outer logic and the exact five scenario laws to current source."""
    scenario_names = np.asarray([item["name"] for item in scenarios])
    scenario_is_s0 = np.asarray(
        [item["is_s0_prefix"] for item in scenarios], dtype=np.bool_
    )
    scenario_nu_sim = np.stack([item["nu_sim"] for item in scenarios])
    scenario_nu_cal = np.stack([item["nu_cal"] for item in scenarios])
    scenario_nu_c_sim = np.asarray(
        [item["nu_c_sim"] for item in scenarios], dtype=np.float64
    )
    scenario_nu_c_cal = np.asarray(
        [item["nu_c_cal"] for item in scenarios], dtype=np.float64
    )
    scenario_laws_sha256 = _hash_named_arrays((
        ("scenario_names", scenario_names),
        ("scenario_is_s0_prefix", scenario_is_s0),
        ("scenario_nu_sim", scenario_nu_sim),
        ("scenario_nu_cal", scenario_nu_cal),
        ("scenario_nu_c_sim", scenario_nu_c_sim),
        ("scenario_nu_c_cal", scenario_nu_c_cal),
    ))
    scenario_law_sha256_by_row = tuple(
        _scenario_law_sha256(item) for item in scenarios
    )
    contract = (
        f"c10b-analysis-contract-v{C10B_ANALYSIS_CONTRACT_VERSION}\n"
        f"C10B_SEED_SCHEMA_VERSION={C10B_SEED_SCHEMA_VERSION}\n"
        + "\n".join(inspect.getsource(function) for function in (
            setup,
            _require_c10_prefix,
            _panel_results_from_prefix,
            _reference_key,
            _run_panel_stream,
            run_scenario,
            _scenario_definitions,
            _scenario_law_sha256,
            main,
        ))
    )
    analysis_sha256 = hashlib.sha256(contract.encode("utf-8")).hexdigest()
    source_sha256 = _sha256_file(Path(__file__).resolve())
    parent = str(c10_parent_stack_sha256)
    stack_sha256 = hashlib.sha256(
        (
            f"c10b-downstream-analysis-stack-v{C10B_ANALYSIS_CONTRACT_VERSION}\n"
            f"{parent}\n{analysis_sha256}\n{scenario_laws_sha256}\n{source_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "c10b_analysis_contract_version": C10B_ANALYSIS_CONTRACT_VERSION,
        "c10b_analysis_contract": contract,
        "c10b_analysis_sha256": analysis_sha256,
        "c10b_scenario_laws_sha256": scenario_laws_sha256,
        "c10b_scenario_law_sha256_by_row": scenario_law_sha256_by_row,
        "c10b_source_sha256": source_sha256,
        "c10b_parent_c10_analysis_stack_sha256": parent,
        "c10b_downstream_analysis_stack_sha256": stack_sha256,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--B-ref", "--B_ref", dest="B_ref", type=int, default=2000)
    ap.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=22)
    ap.add_argument("--fit-seed", type=int, default=c10.FIT_SEED_DEFAULT)
    ap.add_argument("--mc-seed", type=int, default=c10.MC_SEED_DEFAULT)
    args = ap.parse_args()
    if args.N <= 0 or args.B_ref <= 0 or args.n_jobs <= 0:
        ap.error("N, B_ref, and n_jobs must be positive")

    t0 = time.time()
    print("Setup: fit seed is separate from CRN Monte-Carlo seed...")
    setup_result = setup(args.fit_seed)
    prefix = _require_c10_prefix(
        setup_result, args.fit_seed, args.mc_seed, args.N, args.B_ref
    )
    print("  exact c9 DGP identity PASS; exact final-c10 S0 prefix PASS")
    nu_fit = setup_result["nu_null"]
    rho_return = setup_result["rho_return"]
    scenarios = _scenario_definitions(nu_fit)
    c10b_provenance = current_c10b_analysis_provenance(
        scenarios,
        prefix["c10_provenance"]["c10_downstream_analysis_stack_sha256"],
    )

    gaussian_reference = prefix["gaussian"]
    t_reference_cache = {
        _reference_key(nu_fit, float(np.median(nu_fit))): prefix["student_t"]
    }
    outputs = []
    for definition in scenarios:
        outputs.append(run_scenario(
            definition,
            args.N,
            args.B_ref,
            args.n_jobs,
            rho_return,
            prefix["panel_seeds"],
            gaussian_reference,
            t_reference_cache,
            s0_prefix=prefix if definition["is_s0_prefix"] else None,
        ))

    current_c10_end = c10.current_c10_analysis_provenance(
        setup_result["provenance"]["audit_analysis_stack_sha256"]
    )
    if current_c10_end != prefix["c10_provenance"]:
        raise RuntimeError(
            "c10 source/analysis contract changed during c10b computation"
        )
    c10_archive_path = c10.c2.OUT_DIR / Path(prefix["path"]).name
    if _sha256_file(c10_archive_path) != prefix["archive_sha256"]:
        raise RuntimeError("source c10 archive changed during c10b computation")
    c10b_provenance_end = current_c10b_analysis_provenance(
        scenarios,
        prefix["c10_provenance"]["c10_downstream_analysis_stack_sha256"],
    )
    if c10b_provenance_end != c10b_provenance:
        raise RuntimeError(
            "c10b source/analysis contract changed during computation"
        )

    provenance = setup_result["provenance"]
    rows = []
    for scenario_index, output in enumerate(outputs):
        row = dict(output["row"])
        row.update({
            "fit_seed": args.fit_seed,
            "mc_seed": args.mc_seed,
            "rescue_seed": c10.RESCUE_SEED,
            "crn_policy": "same seed positions across all scenarios",
            "design_sha256": provenance["audit_design_sha256"],
            "fixed_null_dgp_sha256": provenance["audit_fixed_null_dgp_sha256"],
            "refitter_sha256": provenance["audit_refitter_sha256"],
            "simulator_sha256": provenance["audit_simulator_sha256"],
            "analysis_stack_sha256": provenance["audit_analysis_stack_sha256"],
            "source_c10_archive_sha256": prefix["archive_sha256"],
            "source_c10_analysis_sha256": prefix["c10_provenance"][
                "c10_analysis_sha256"
            ],
            "source_c10_source_sha256": prefix["c10_provenance"][
                "c10_source_sha256"
            ],
            "source_c10_downstream_analysis_stack_sha256": prefix[
                "c10_provenance"
            ]["c10_downstream_analysis_stack_sha256"],
            "c10b_analysis_contract_version": c10b_provenance[
                "c10b_analysis_contract_version"
            ],
            "c10b_analysis_sha256": c10b_provenance["c10b_analysis_sha256"],
            "c10b_scenario_laws_sha256": c10b_provenance[
                "c10b_scenario_laws_sha256"
            ],
            "scenario_law_sha256": c10b_provenance[
                "c10b_scenario_law_sha256_by_row"
            ][scenario_index],
            "c10b_source_sha256": c10b_provenance["c10b_source_sha256"],
            "c10b_parent_c10_analysis_stack_sha256": c10b_provenance[
                "c10b_parent_c10_analysis_stack_sha256"
            ],
            "c10b_downstream_analysis_stack_sha256": c10b_provenance[
                "c10b_downstream_analysis_stack_sha256"
            ],
        })
        rows.append(row)
    out_csv = c10.c2.OUT_DIR / "c10b-dgp-perturbation-results.csv"
    out_csv_tmp = out_csv.with_name(out_csv.stem + ".write-tmp.csv")
    pd.DataFrame(rows).to_csv(out_csv_tmp, index=False)

    panel_usable = np.stack([o["panel"]["panel_usable_mask"] for o in outputs])
    panel_se_ok = np.stack([o["panel"]["panel_se_ok_full"] for o in outputs])
    panel_di = np.stack([o["panel"]["delta_infra_full"] for o in outputs])
    panel_dr = np.stack([o["panel"]["delta_reg_full"] for o in outputs])
    panel_sei = np.stack([o["panel"]["se_infra_full"] for o in outputs])
    panel_ser = np.stack([o["panel"]["se_reg_full"] for o in outputs])
    panel_dbar = np.stack([o["panel"]["dbar_full"] for o in outputs])
    panel_p_iid = np.stack([o["panel"]["p_iid_full"] for o in outputs])
    panel_p_de = np.stack([o["panel"]["p_de_full"] for o in outputs])
    t_ref_full = np.stack([o["t_reference"]["full"] for o in outputs])
    t_ref_mask = np.stack([o["t_reference"]["usable_mask"] for o in outputs])
    g_ref_full = gaussian_reference["full"]
    g_ref_mask = gaussian_reference["usable_mask"]
    audit = setup_result["audit_payload"]
    out_npz = c10.c2.OUT_DIR / "c10b-dgp-perturbation-draws.npz"
    tmp_npz = out_npz.with_name(out_npz.stem + ".write-tmp.npz")
    np.savez(
        tmp_npz,
        scenario_names=np.asarray([o["name"] for o in outputs]),
        scenario_is_s0_prefix=np.asarray([o["is_s0_prefix"] for o in outputs], dtype=bool),
        scenario_nu_sim=np.stack([o["nu_sim"] for o in outputs]),
        scenario_nu_cal=np.stack([o["nu_cal"] for o in outputs]),
        scenario_nu_c_sim=np.asarray([o["nu_c_sim"] for o in outputs], dtype=float),
        scenario_nu_c_cal=np.asarray([o["nu_c_cal"] for o in outputs], dtype=float),
        scenario_law_sha256=np.asarray(
            c10b_provenance["c10b_scenario_law_sha256_by_row"]
        ),
        panel_seeds=prefix["panel_seeds"],
        scenario_panel_seeds=np.tile(prefix["panel_seeds"], (len(outputs), 1)),
        scenario_panel_usable_mask=panel_usable,
        scenario_panel_excluded_mask=~panel_usable,
        scenario_panel_se_ok_full=panel_se_ok,
        scenario_panel_delta_infra_full=panel_di,
        scenario_panel_delta_reg_full=panel_dr,
        scenario_panel_se_infra_full=panel_sei,
        scenario_panel_se_reg_full=panel_ser,
        scenario_panel_dbar_full=panel_dbar,
        scenario_panel_p_iid_full=panel_p_iid,
        scenario_panel_p_de_full=panel_p_de,
        gaussian_reference_seeds=gaussian_reference["seeds"],
        gaussian_reference_full=g_ref_full,
        gaussian_reference_usable_mask=g_ref_mask,
        gaussian_reference_excluded_mask=~g_ref_mask,
        scenario_gaussian_reference_seeds=np.tile(
            gaussian_reference["seeds"], (len(outputs), 1)
        ),
        scenario_gaussian_reference_full=np.tile(g_ref_full, (len(outputs), 1)),
        scenario_gaussian_reference_usable_mask=np.tile(g_ref_mask, (len(outputs), 1)),
        scenario_gaussian_reference_excluded_mask=np.tile(~g_ref_mask, (len(outputs), 1)),
        t_reference_seeds=prefix["student_t"]["seeds"],
        scenario_t_reference_seeds=np.tile(prefix["student_t"]["seeds"], (len(outputs), 1)),
        scenario_t_reference_full=t_ref_full,
        scenario_t_reference_usable_mask=t_ref_mask,
        scenario_t_reference_excluded_mask=~t_ref_mask,
        scenario_crit_gauss=np.asarray([
            [o["crit"]["gauss"][0.05], o["crit"]["gauss"][0.10]] for o in outputs
        ]),
        scenario_crit_t=np.asarray([
            [o["crit"]["t"][0.05], o["crit"]["t"][0.10]] for o in outputs
        ]),
        c10b_seed_schema_version=np.int64(C10B_SEED_SCHEMA_VERSION),
        c10b_seed_schema=np.asarray(
            "fit_seed defines observed multistart only; all scenarios use identical "
            "panel/reference seed positions as common random numbers; S0 is the exact "
            "N/B_ref prefix of final c10; Gaussian reference is cached once"
        ),
        fit_seed=np.int64(args.fit_seed),
        mc_seed=np.int64(args.mc_seed),
        rescue_seed=np.int64(c10.RESCUE_SEED),
        panel_first_seed=np.int64(prefix["panel_seeds"][0]),
        panel_last_seed=np.int64(prefix["panel_seeds"][-1]),
        gaussian_reference_first_seed=np.int64(gaussian_reference["seeds"][0]),
        gaussian_reference_last_seed=np.int64(gaussian_reference["seeds"][-1]),
        t_reference_first_seed=np.int64(prefix["student_t"]["seeds"][0]),
        t_reference_last_seed=np.int64(prefix["student_t"]["seeds"][-1]),
        N_requested=np.int64(args.N),
        B_ref_requested=np.int64(args.B_ref),
        asset_names=np.asarray(ASSETS),
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
        null_params_by_asset=audit["audit_null_params_by_asset"],
        null_mean_returns=audit["audit_null_mean_returns"],
        null_sigma2_concat=audit["audit_null_sigma2_concat"],
        null_sigma2_offsets=audit["audit_null_sigma2_offsets"],
        nu_null_by_asset=audit["audit_nu_null_by_asset"],
        nu_c_null=audit["audit_nu_c_null"],
        R_z=audit["audit_R_z"],
        L_z=audit["audit_L_z"],
        audit_design_sha256=audit["audit_design_sha256"],
        audit_fixed_null_dgp_sha256=audit["audit_fixed_null_dgp_sha256"],
        audit_refitter_sha256=audit["audit_refitter_sha256"],
        audit_simulator_sha256=audit["audit_simulator_sha256"],
        audit_analysis_stack_sha256=audit["audit_analysis_stack_sha256"],
        c9_seed_schema_version=np.int64(2),
        c9_exact_identity_verified=np.bool_(True),
        c10_s0_prefix_verified=np.bool_(True),
        c10_source_archive=np.asarray(prefix["path"]),
        c10_source_archive_sha256=np.asarray(prefix["archive_sha256"]),
        source_c10_analysis_contract_version=np.int64(
            prefix["c10_provenance"]["c10_analysis_contract_version"]
        ),
        source_c10_analysis_sha256=np.asarray(
            prefix["c10_provenance"]["c10_analysis_sha256"]
        ),
        source_c10_source_sha256=np.asarray(
            prefix["c10_provenance"]["c10_source_sha256"]
        ),
        source_c10_parent_c9_analysis_stack_sha256=np.asarray(
            prefix["c10_provenance"]["c10_parent_c9_analysis_stack_sha256"]
        ),
        source_c10_downstream_analysis_stack_sha256=np.asarray(
            prefix["c10_provenance"]["c10_downstream_analysis_stack_sha256"]
        ),
        c10b_analysis_contract_version=np.int64(
            c10b_provenance["c10b_analysis_contract_version"]
        ),
        c10b_analysis_contract=np.asarray(
            c10b_provenance["c10b_analysis_contract"]
        ),
        c10b_analysis_sha256=np.asarray(c10b_provenance["c10b_analysis_sha256"]),
        c10b_scenario_laws_sha256=np.asarray(
            c10b_provenance["c10b_scenario_laws_sha256"]
        ),
        c10b_source_sha256=np.asarray(c10b_provenance["c10b_source_sha256"]),
        c10b_parent_c10_analysis_stack_sha256=np.asarray(
            c10b_provenance["c10b_parent_c10_analysis_stack_sha256"]
        ),
        c10b_downstream_analysis_stack_sha256=np.asarray(
            c10b_provenance["c10b_downstream_analysis_stack_sha256"]
        ),
    )
    with np.load(tmp_npz, allow_pickle=False) as trial:
        expected_panel_shape = (len(outputs), args.N)
        expected_reference_shape = (len(outputs), args.B_ref)
        if trial["scenario_panel_dbar_full"].shape != expected_panel_shape:
            raise RuntimeError("c10b temporary archive lost the scenario panel shape")
        if trial["scenario_t_reference_full"].shape != expected_reference_shape:
            raise RuntimeError("c10b temporary archive lost the t-reference shape")
        if not np.array_equal(
            trial["scenario_panel_usable_mask"],
            ~np.isnan(trial["scenario_panel_dbar_full"]),
        ):
            raise RuntimeError("c10b temporary archive panel mask/value mismatch")
        if not np.array_equal(
            trial["scenario_t_reference_usable_mask"],
            ~np.isnan(trial["scenario_t_reference_full"]),
        ):
            raise RuntimeError("c10b temporary archive t-reference mask/value mismatch")
        if not np.array_equal(
            trial["scenario_panel_dbar_full"][0], prefix["dbar_full"], equal_nan=True
        ):
            raise RuntimeError("c10b temporary archive lost exact c10 S0 panel prefix")
        if not np.array_equal(
            trial["scenario_t_reference_full"][0],
            prefix["student_t"]["full"],
            equal_nan=True,
        ):
            raise RuntimeError("c10b temporary archive lost exact c10 S0 reference prefix")
        if not bool(trial["c9_exact_identity_verified"]) or not bool(
            trial["c10_s0_prefix_verified"]
        ):
            raise RuntimeError("c10b temporary archive lost a provenance gate")
        trial_laws_sha256 = _hash_named_arrays((
            ("scenario_names", trial["scenario_names"]),
            ("scenario_is_s0_prefix", trial["scenario_is_s0_prefix"]),
            ("scenario_nu_sim", trial["scenario_nu_sim"]),
            ("scenario_nu_cal", trial["scenario_nu_cal"]),
            ("scenario_nu_c_sim", trial["scenario_nu_c_sim"]),
            ("scenario_nu_c_cal", trial["scenario_nu_c_cal"]),
        ))
        if trial_laws_sha256 != c10b_provenance["c10b_scenario_laws_sha256"]:
            raise RuntimeError("c10b temporary archive changed a scenario law")
        trial_row_hashes = []
        for scenario_index in range(len(outputs)):
            trial_row_hashes.append(_scenario_law_sha256({
                "name": str(trial["scenario_names"][scenario_index]),
                "is_s0_prefix": bool(
                    trial["scenario_is_s0_prefix"][scenario_index]
                ),
                "nu_sim": trial["scenario_nu_sim"][scenario_index],
                "nu_cal": trial["scenario_nu_cal"][scenario_index],
                "nu_c_sim": float(trial["scenario_nu_c_sim"][scenario_index]),
                "nu_c_cal": float(trial["scenario_nu_c_cal"][scenario_index]),
            }))
        if not np.array_equal(
            trial["scenario_law_sha256"], np.asarray(trial_row_hashes)
        ) or tuple(trial_row_hashes) != tuple(
            c10b_provenance["c10b_scenario_law_sha256_by_row"]
        ):
            raise RuntimeError("c10b temporary archive changed a row scenario hash")
        for key in (
            "c10b_analysis_contract",
            "c10b_analysis_sha256",
            "c10b_scenario_laws_sha256",
            "c10b_source_sha256",
            "c10b_parent_c10_analysis_stack_sha256",
            "c10b_downstream_analysis_stack_sha256",
        ):
            if c10._scalar_text(trial[key]) != str(c10b_provenance[key]):
                raise RuntimeError(f"c10b temporary archive changed {key}")
        if c10._scalar_text(trial["c10_source_archive_sha256"]) != str(
            prefix["archive_sha256"]
        ):
            raise RuntimeError("c10b temporary archive changed source-c10 hash")
        if c10._scalar_text(
            trial["source_c10_downstream_analysis_stack_sha256"]
        ) != str(
            prefix["c10_provenance"]["c10_downstream_analysis_stack_sha256"]
        ):
            raise RuntimeError("c10b temporary archive changed parent c10 stack")
    csv_trial = pd.read_csv(out_csv_tmp)
    if csv_trial["scenario"].tolist() != [item["name"] for item in scenarios]:
        raise RuntimeError("c10b CSV changed scenario order")
    for key in (
        "c10b_analysis_sha256",
        "c10b_scenario_laws_sha256",
        "c10b_source_sha256",
        "c10b_parent_c10_analysis_stack_sha256",
        "c10b_downstream_analysis_stack_sha256",
    ):
        if set(csv_trial[key].astype(str)) != {str(c10b_provenance[key])}:
            raise RuntimeError(f"c10b CSV changed {key}")
    if csv_trial["scenario_law_sha256"].astype(str).tolist() != list(
        c10b_provenance["c10b_scenario_law_sha256_by_row"]
    ):
        raise RuntimeError("c10b CSV changed a row scenario-law hash")
    finding_path = c10.c2.OUT_DIR / "c10b-dgp-perturbation-FINDING.md"
    finding_tmp = finding_path.with_name(finding_path.stem + ".write-tmp.md")
    finding_tmp.write_text(
        _finding_text(rows, args.N, args.B_ref, args.fit_seed, args.mc_seed)
    )
    finding_trial = finding_tmp.read_text()
    if "# C10b -- DGP-perturbation sensitivity" not in finding_trial:
        raise RuntimeError("c10b temporary finding failed validation")
    tmp_npz.replace(out_npz)
    out_csv_tmp.replace(out_csv)
    finding_tmp.replace(finding_path)
    print(f"\nSaved results CSV: {out_csv}")
    print(f"Saved full CRN archive: {out_npz}")
    print(f"Saved finding: {finding_path}")
    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
