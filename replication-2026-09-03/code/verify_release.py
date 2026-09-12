#!/usr/bin/env python3
"""Fast consistency checks for the accepted-paper replication package.

This verifier does not rerun the computationally intensive bootstraps. It checks
that the committed source data, result artifacts, and manuscript agree on the
decision-relevant values produced by those runs.
"""

from pathlib import Path
import hashlib
import inspect
import io
import json
import math
import re
import numpy as np
import pandas as pd
from scipy import stats

import c2_relaxed_threshold_sensitivity as c2
import c7_ccc_garchx_bootstrap as c7
import c9_tcopula_bootstrap as c9
import c10_size_study as c10
import c10b_dgp_perturbation as c10b
import c13_rung4_recompute as c13
import c19_recursive_sensitivity as c19
import c20_nuc_sensitivity as c20
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA, _variance_recursion_core
from verify_tables import verify_tables


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
RESULTS = ROOT / "results"
ASSETS = np.asarray(["btc", "eth", "xrp", "bnb", "ltc", "ada"])
SPECS = ("baseline", "crisis", "full")
FIT_SEED = 12345
DRAW_SEED_OFFSET = 10_000
MC_SEED = 20260618
RESCUE_SEED = 20260807
HASH_KEYS = (
    "audit_design_sha256",
    "audit_fixed_null_dgp_sha256",
    "audit_refitter_sha256",
    "audit_simulator_sha256",
    "audit_analysis_stack_sha256",
)
HASH_RE = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(value, target, tol=5e-6):
    return math.isclose(float(value), target, rel_tol=0.0, abs_tol=tol)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothed_tail(hits, used):
    return (int(hits) + 1) / (int(used) + 1)


def load_npz(path):
    path = Path(path)
    require(path.is_file(), f"required artifact is missing: {path.relative_to(ROOT)}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def scalar(payload, key):
    require(key in payload, f"archive lacks required field: {key}")
    value = np.asarray(payload[key])
    require(value.shape == (), f"{key} must be scalar, got shape {value.shape}")
    return value.item()


def scalar_text(payload, key):
    return str(scalar(payload, key))


def require_fields(container, required, label):
    available = set(container.files if hasattr(container, "files") else container)
    missing = sorted(set(required).difference(available))
    require(not missing, f"{label} uses a stale/incomplete schema: missing {missing}")


def require_exact(actual, expected, message):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    require(actual.shape == expected.shape, f"{message}: shape {actual.shape} != {expected.shape}")
    if actual.dtype.kind in "fc" or expected.dtype.kind in "fc":
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    require(equal, message)


def require_near(actual, expected, message, tol=1e-12):
    require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol),
        f"{message}: {float(actual)!r} != {float(expected)!r}",
    )


def require_hash(value, label):
    text = str(value)
    require(HASH_RE.fullmatch(text) is not None, f"{label} is not a SHA-256 digest")
    return text


def hash_named_arrays(named_arrays):
    """Recompute c9's platform-stable digest from archived arrays."""
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


def indexed_rows(frame, key, expected, label):
    require(key in frame.columns, f"{label} lacks index column {key!r}")
    require(not frame[key].duplicated().any(), f"{label} contains duplicate {key} rows")
    actual = frame[key].astype(str).tolist()
    require(actual == list(expected), f"{label} row order/names drifted: {actual}")
    return frame.set_index(key, drop=False)


def _csv_int(row, field, expected, label):
    require(field in row.index, f"{label} CSV lacks {field}")
    require(int(row[field]) == int(expected), f"{label} CSV {field} is inconsistent")


def _csv_float(row, field, expected, label, tol=1e-12):
    require(field in row.index, f"{label} CSV lacks {field}")
    require_near(row[field], expected, f"{label} CSV {field} is inconsistent", tol=tol)


def require_finding(path, heading, snippets, label):
    """Require a generated finding and its decision-relevant rendered values."""
    path = Path(path)
    require(path.is_file(), f"{label} finding is missing")
    text = path.read_text()
    require(heading in text, f"{label} finding heading drifted")
    for snippet_label, snippet in snippets:
        require(snippet in text, f"{label} finding lacks current {snippet_label}")
    return text


def require_current_provenance(payload, expected, label):
    """Match scalar archived provenance to a current-source reconstruction."""
    require_fields(payload, expected, label)
    for key, expected_value in expected.items():
        actual = scalar(payload, key)
        if isinstance(expected_value, (int, np.integer)):
            require(int(actual) == int(expected_value),
                    f"{label} {key} version/value drifted")
        else:
            require(str(actual) == str(expected_value),
                    f"{label} {key} does not match current packaged source")
        if key.endswith("sha256"):
            require_hash(str(actual), f"{label} {key}")


def _current_c9_contracts():
    """Reconstruct c9's fingerprints from the implementation being verified."""
    refitter_contract = (
        f"assets={','.join(c7.ASSETS)};MAX_ITER={c7.MAX_ITER};"
        f"DELTA_CAP={c7.DELTA_CAP:.17g};rescue_starts=6;"
        f"rescue_seed={c7._GLOBAL.get('rescue_seed', RESCUE_SEED)};"
        f"numba_enabled={_HAVE_NUMBA}\n"
        + inspect.getsource(c7._plausible_delta)
        + inspect.getsource(c7._refit_unrestricted_dbar)
        + inspect.getsource(_variance_recursion_core)
        + inspect.getsource(FastTARCHX)
    )
    simulator_contract = (
        "seed_schema_version=2;one_union_calendar_by_six_latent_draw_per_seed\n"
        + inspect.getsource(c7.install_simulation_calendar)
        + inspect.getsource(c7.scatter_calendar_innovations)
        + inspect.getsource(c7.draw_gaussian_calendar_innovations)
        + inspect.getsource(c7._draw_parametric_null)
        + inspect.getsource(c9._student_t_innovations)
        + inspect.getsource(c9._student_t_calendar_innovations)
        + inspect.getsource(c9._draw_parametric_null_t_core)
    )
    return refitter_contract, simulator_contract


def _current_c19_contract():
    """Reconstruct c19's recursive contract from current source and constants."""
    return (
        f"recursive_seed_schema_version={c19.RECURSIVE_SEED_SCHEMA_VERSION};"
        f"draw_seed_offset={c19.DRAW_SEED_OFFSET};"
        f"SIG2_FLOOR={c19.SIG2_FLOOR:.17g};"
        f"SIG2_EXPLODE={c19.SIG2_EXPLODE:.17g};"
        f"p_value_contract={c19.P_VALUE_CONTRACT}\n"
        + inspect.getsource(c19._simulate_recursive_panel)
        + inspect.getsource(c19._draw_recursive_null_t)
    )


def validate_c3_c8e():
    """Validate the structural-break partition and its C8e consumer.

    C3 stores inclusive segment endpoints even though the producer constructs
    the segments as positional half-open slices.  These checks bind those
    endpoints back to the actual input calendars, so a duplicated break-date
    observation or an omitted final observation cannot pass release review.
    """
    breaks_path = RESULTS / "c3-bai-perron-results.csv"
    persistence_path = RESULTS / "c3-subsample-persistence.csv"
    summary_path = RESULTS / "c3-break-summary.md"
    c8e_path = RESULTS / "c8e-persistence-se.csv"
    for path in (breaks_path, persistence_path, summary_path, c8e_path):
        require(path.is_file(), f"required artifact is missing: {path.relative_to(ROOT)}")

    breaks = pd.read_csv(breaks_path, float_precision="round_trip")
    persistence = pd.read_csv(persistence_path, float_precision="round_trip")
    c8e_frame = pd.read_csv(c8e_path, float_precision="round_trip")
    require_fields(
        breaks.columns,
        {
            "asset", "series_type", "break_number", "break_date",
            "ci_lo_date", "ci_hi_date", "pre_mean", "post_mean",
            "delta_mean", "t_stat", "p_value", "pre_n", "post_n",
        },
        "c3 break results CSV",
    )
    require_fields(
        persistence.columns,
        {
            "asset", "subsample_start", "subsample_end", "n_obs",
            "alpha", "beta", "persistence", "log_lik",
            "full_sample_persistence",
        },
        "c3 persistence CSV",
    )
    require_fields(
        c8e_frame.columns,
        {
            "asset", "segment", "start", "end", "n_obs", "alpha",
            "se_alpha", "beta", "se_beta", "gamma", "se_gamma",
            "persistence", "se_persistence", "small_sample_n_lt_500",
            "full_persistence", "drop_vs_full", "drop_exceeds_2se",
        },
        "c8e persistence CSV",
    )

    require(len(breaks) == 44, f"c3 must contain 44 break rows, found {len(breaks)}")
    require(len(persistence) == 27,
            f"c3 must contain 27 persistence segments, found {len(persistence)}")
    require(not breaks.duplicated(["asset", "series_type", "break_number"]).any(),
            "c3 contains a duplicate asset/series/break-number key")
    require(not breaks.duplicated(["asset", "series_type", "break_date"]).any(),
            "c3 contains a duplicate asset/series/break-date key")
    require(set(breaks["asset"].astype(str)) == set(ASSETS),
            "c3 break-result asset coverage drifted")
    require(set(breaks["series_type"].astype(str)) == {"abs_return", "cond_variance"},
            "c3 break-result series types drifted")

    panels = c2.load_returns_panel()
    require(set(panels) == set(ASSETS), "c3 input-panel asset coverage drifted")
    expected_segment_sizes = {
        "btc": [693, 612, 1129],
        "eth": [673, 365, 372, 471, 553],
        "xrp": [682, 365, 369, 594, 424],
        "bnb": [490, 328, 390, 354, 629],
        "ltc": [674, 387, 365, 459, 549],
        "ada": [689, 582, 527, 636],
    }

    for asset in ASSETS:
        series = panels[str(asset)].dropna()
        index = pd.DatetimeIndex(series.index)
        require(index.is_monotonic_increasing and index.is_unique,
                f"c3 {asset} input calendar is not strictly ordered and unique")
        n_total = len(index)
        for series_type in ("abs_return", "cond_variance"):
            rows = breaks.loc[
                (breaks["asset"] == asset) & (breaks["series_type"] == series_type)
            ].copy()
            rows = rows.sort_values("break_number", kind="stable")
            expected_numbers = list(range(1, len(rows) + 1))
            require(rows["break_number"].astype(int).tolist() == expected_numbers,
                    f"c3 {asset} {series_type} break numbers are not consecutive")
            prior_pre_n = 0
            for row in rows.itertuples(index=False):
                pre_n = int(row.pre_n)
                post_n = int(row.post_n)
                require(0 < pre_n < n_total and pre_n + post_n == n_total,
                        f"c3 {asset} {series_type} pre/post counts do not partition N")
                require(pre_n > prior_pre_n,
                        f"c3 {asset} {series_type} break positions are not increasing")
                prior_pre_n = pre_n
                break_date = pd.Timestamp(row.break_date)
                ci_lo = pd.Timestamp(row.ci_lo_date)
                ci_hi = pd.Timestamp(row.ci_hi_date)
                require(break_date == index[pre_n],
                        f"c3 {asset} {series_type} break date is not the first post-break observation")
                require(ci_lo in index and ci_hi in index,
                        f"c3 {asset} {series_type} confidence interval is off-calendar")
                require(ci_lo <= break_date <= ci_hi,
                        f"c3 {asset} {series_type} confidence interval does not contain its break")
                require_near(
                    row.delta_mean,
                    float(row.post_mean) - float(row.pre_mean),
                    f"c3 {asset} {series_type} delta is not post mean minus pre mean",
                    tol=2e-12,
                )
                require(np.isfinite([row.pre_mean, row.post_mean, row.delta_mean,
                                     row.t_stat, row.p_value]).all(),
                        f"c3 {asset} {series_type} contains non-finite break statistics")
                require(0.0 <= float(row.p_value) <= 1.0,
                        f"c3 {asset} {series_type} p-value is outside [0,1]")
                if series_type == "abs_return":
                    values = series.abs().to_numpy(dtype=float)
                    require_near(row.pre_mean, values[:pre_n].mean(),
                                 f"c3 {asset} absolute-return pre mean drifted")
                    require_near(row.post_mean, values[pre_n:].mean(),
                                 f"c3 {asset} absolute-return post mean drifted")
                    t_stat, p_value = stats.ttest_ind(
                        values[pre_n:], values[:pre_n], equal_var=False
                    )
                    require_near(row.t_stat, t_stat,
                                 f"c3 {asset} absolute-return Welch statistic drifted")
                    require_near(row.p_value, p_value,
                                 f"c3 {asset} absolute-return Welch p-value drifted")

    require(not persistence.duplicated(["asset", "subsample_start", "subsample_end"]).any(),
            "c3 persistence CSV contains duplicate segment keys")
    require(set(persistence["asset"].astype(str)) == set(ASSETS),
            "c3 persistence asset coverage drifted")
    require(np.isfinite(persistence[["alpha", "beta", "persistence", "log_lik",
                                     "full_sample_persistence"]].to_numpy(dtype=float)).all(),
            "c3 persistence CSV contains non-finite estimates")
    require(np.allclose(
        persistence["persistence"].to_numpy(dtype=float),
        (persistence["alpha"] + persistence["beta"]).to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-15,
    ), "c3 persistence is not alpha + beta")

    expected_c8e_segments = []
    for asset in ASSETS:
        series = panels[str(asset)].dropna()
        index = pd.DatetimeIndex(series.index)
        asset_breaks = breaks.loc[
            (breaks["asset"] == asset) & (breaks["series_type"] == "abs_return")
        ].sort_values("break_number", kind="stable")
        break_positions = asset_breaks["pre_n"].astype(int).tolist()
        expected_bounds = [0, *break_positions, len(index)]
        rows = persistence.loc[persistence["asset"] == asset].copy()
        require(len(rows) == len(expected_bounds) - 1,
                f"c3 {asset} persistence segment count is not abs-return breaks + 1")
        expected_starts = [index[lo] for lo in expected_bounds[:-1]]
        expected_ends = [index[hi - 1] for hi in expected_bounds[1:]]
        actual_starts = pd.to_datetime(rows["subsample_start"], errors="raise").tolist()
        actual_ends = pd.to_datetime(rows["subsample_end"], errors="raise").tolist()
        require(actual_starts == expected_starts and actual_ends == expected_ends,
                f"c3 {asset} inclusive endpoints do not match the exact break partition")
        expected_n = np.diff(np.asarray(expected_bounds, dtype=int))
        require_exact(rows["n_obs"].to_numpy(dtype=int), expected_n,
                      f"c3 {asset} segment counts do not match inclusive endpoints")
        require_exact(rows["n_obs"].to_numpy(dtype=int), expected_segment_sizes[str(asset)],
                      f"c3 {asset} canonical segment sizes drifted")
        require(int(rows["n_obs"].sum()) == len(index),
                f"c3 {asset} persistence segments do not cover every input observation once")
        full_values = rows["full_sample_persistence"].to_numpy(dtype=float)
        require(np.array_equal(full_values, np.repeat(full_values[0], len(full_values))),
                f"c3 {asset} full-sample persistence is not constant across segments")
        for segment_number, row in enumerate(rows.itertuples(index=False)):
            expected_c8e_segments.append((
                str(asset), f"seg{segment_number}", str(row.subsample_start),
                str(row.subsample_end), int(row.n_obs),
            ))

    # C8e is a different GJR diagnostic, but it must consume exactly C3's
    # segment surface.  Its estimates are therefore checked for internal
    # consistency without equating them to C3's symmetric-GARCH estimates.
    require(len(c8e_frame) == 33,
            f"c8e must contain 6 FULL + 27 segment rows, found {len(c8e_frame)}")
    require(not c8e_frame.duplicated(["asset", "segment"]).any(),
            "c8e contains duplicate asset/segment keys")
    full = c8e_frame.loc[c8e_frame["segment"] == "FULL"].copy()
    subs = c8e_frame.loc[c8e_frame["segment"] != "FULL"].copy()
    require(len(full) == 6 and set(full["asset"].astype(str)) == set(ASSETS),
            "c8e must contain exactly one FULL row for each asset")
    require(len(subs) == 27, "c8e must contain exactly 27 non-FULL rows")
    actual_c8e_segments = [
        (str(row.asset), str(row.segment), str(row.start), str(row.end), int(row.n_obs))
        for row in subs.itertuples(index=False)
    ]
    require(actual_c8e_segments == expected_c8e_segments,
            "c8e non-FULL rows do not exactly match C3's ordered segment surface")
    require(np.allclose(
        c8e_frame["persistence"].to_numpy(dtype=float),
        (c8e_frame["alpha"] + c8e_frame["beta"] + c8e_frame["gamma"] / 2.0).to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-15,
    ), "c8e GJR persistence is not alpha + beta + gamma/2")
    expected_se = np.sqrt(
        c8e_frame["se_alpha"].to_numpy(dtype=float) ** 2
        + c8e_frame["se_beta"].to_numpy(dtype=float) ** 2
        + c8e_frame["se_gamma"].to_numpy(dtype=float) ** 2 / 4.0
    )
    require(np.allclose(c8e_frame["se_persistence"].to_numpy(dtype=float), expected_se,
                        rtol=0.0, atol=2e-15, equal_nan=True),
            "c8e persistence SE does not match its documented diagonal approximation")
    expected_small = c8e_frame["n_obs"].to_numpy(dtype=int) < 500
    require_exact(c8e_frame["small_sample_n_lt_500"].to_numpy(dtype=bool), expected_small,
                  "c8e small-sample flags drifted")
    for asset in ASSETS:
        source_index = pd.DatetimeIndex(panels[str(asset)].dropna().index)
        full_row = full.loc[full["asset"] == asset].iloc[0]
        asset_subs = subs.loc[subs["asset"] == asset]
        require(int(full_row["n_obs"]) == int(asset_subs["n_obs"].sum()) == len(source_index),
                f"c8e {asset} FULL count does not equal its C3 partition total")
        require(pd.Timestamp(full_row["start"]) == source_index[0] and
                pd.Timestamp(full_row["end"]) == source_index[-1],
                f"c8e {asset} FULL endpoints do not match the input calendar")
        require(np.allclose(
            asset_subs["full_persistence"].to_numpy(dtype=float),
            float(full_row["persistence"]), rtol=0.0, atol=2e-15,
        ), f"c8e {asset} segment rows do not carry the FULL-row persistence")
        require(np.allclose(
            asset_subs["drop_vs_full"].to_numpy(dtype=float),
            float(full_row["persistence"]) - asset_subs["persistence"].to_numpy(dtype=float),
            rtol=0.0, atol=2e-15,
        ), f"c8e {asset} persistence drops are inconsistent")

    mean_by_asset = persistence.groupby("asset", sort=False)["persistence"].mean()
    full_by_asset = persistence.groupby("asset", sort=False)["full_sample_persistence"].first()
    mean_sub = float(mean_by_asset.reindex(ASSETS).mean())
    mean_full = float(full_by_asset.reindex(ASSETS).mean())
    summary = summary_path.read_text()
    require(
        f"Equal-weight mean full-sample persistence (α + β): **{mean_full:.4f}**" in summary,
        "c3 summary does not render the current equal-weight full-sample mean",
    )
    require(
        "Equal-weight mean of the six assets' within-segment averages: "
        f"**{mean_sub:.4f}**" in summary,
        "c3 summary does not render the current equal-weight within-segment mean",
    )
    require(
        f"Difference: **{mean_full - mean_sub:+.4f}** (within-segment point estimates lower)"
        in summary,
        "c3 summary does not render the current descriptive persistence difference",
    )

    manuscript = (ROOT / "main.tex").read_text()
    label = r"\label{tab:bai_perron_persistence}"
    label_pos = manuscript.find(label)
    require(label_pos >= 0, "main manuscript lacks the C3 persistence table label")
    table_start = manuscript.rfind(r"\begin{table}", 0, label_pos)
    table_end = manuscript.find(r"\end{table}", label_pos)
    require(table_start >= 0 and table_end >= 0,
            "could not isolate the C3 persistence table in main.tex")
    section_start = manuscript.rfind(r"\subsubsection{Persistence", 0, table_start)
    require(section_start >= 0, "could not isolate the C3 persistence prose in main.tex")
    persistence_prose = manuscript[section_start:table_start]
    require(re.search(rf"\\alpha\+\\beta\\approx\s*{mean_sub:.3f}", persistence_prose),
            "main C3 persistence prose has a stale within-segment mean")
    require(re.search(rf"full-sample(?: mean of)?\s+\${mean_full:.3f}\$", persistence_prose),
            "main C3 persistence prose has a stale full-sample mean")
    table_text = manuscript[table_start:table_end]
    for asset in ASSETS:
        full_value = float(full_by_asset.loc[asset])
        sub_value = float(mean_by_asset.loc[asset])
        n_segments = int((persistence["asset"] == asset).sum())
        rendered = re.compile(
            rf"(?m)^{str(asset).upper()}\s*&\s*{full_value:.3f}\s*&\s*"
            rf"{sub_value:.3f}\s*&\s*{n_segments}\s*\\\\\s*$"
        )
        require(rendered.search(table_text) is not None,
                f"main C3 persistence table row is stale for {asset}")
    mean_row = re.compile(
        rf"(?m)^\\textbf\{{Mean\}}\s*&\s*\\textbf\{{{mean_full:.3f}\}}\s*&\s*"
        rf"\\textbf\{{{mean_sub:.3f}\}}\s*&\s*--\s*\\\\\s*$"
    )
    require(mean_row.search(table_text) is not None,
            "main C3 persistence table mean row is stale")
    return breaks, persistence, c8e_frame


def _validate_union_calendar(payload, prefix, label):
    """Validate the exact 2,434-date union-calendar/availability contract."""
    def key(name):
        return prefix + name

    required = {
        key("asset_names"), key("simulation_calendar_ns"),
        key("calendar_pos_concat"), key("calendar_pos_offsets"),
        key("availability_matrix"), key("availability_pattern_code"),
        key("availability_pattern_codes"), key("availability_pattern_counts"),
        key("availability_pattern_first_date_ns"),
        key("availability_pattern_last_date_ns"), key("R_z"), key("L_z"),
    }
    require_fields(payload, required, label)
    require_exact(payload[key("asset_names")], ASSETS, f"{label} asset order drifted")
    calendar = np.asarray(payload[key("simulation_calendar_ns")], dtype=np.int64)
    positions = np.asarray(payload[key("calendar_pos_concat")], dtype=np.int64)
    offsets = np.asarray(payload[key("calendar_pos_offsets")], dtype=np.int64)
    availability = np.asarray(payload[key("availability_matrix")], dtype=bool)
    require(calendar.shape == (2434,), f"{label} union calendar must contain 2,434 dates")
    require(np.all(np.diff(calendar) > 0), f"{label} union calendar is not strictly increasing")
    require(offsets.shape == (7,) and offsets[0] == 0 and offsets[-1] == len(positions),
            f"{label} calendar-position offsets are malformed")
    require(np.all(np.diff(offsets) >= 0), f"{label} calendar-position offsets decrease")
    require(availability.shape == (2434, 6), f"{label} availability matrix shape drifted")
    for j, asset in enumerate(ASSETS):
        pos = positions[offsets[j]:offsets[j + 1]]
        require(np.all((0 <= pos) & (pos < len(calendar))),
                f"{label} {asset} calendar positions are out of bounds")
        require(len(np.unique(pos)) == len(pos),
                f"{label} {asset} calendar positions contain duplicates")
        implied = np.zeros(len(calendar), dtype=bool)
        implied[pos] = True
        require_exact(availability[:, j], implied,
                      f"{label} {asset} positions and availability mask disagree")
    require_exact(
        availability.sum(axis=0),
        np.asarray([2434, 2434, 2434, 2191, 2434, 2434]),
        f"{label} per-asset calendar coverage drifted",
    )
    weights = 1 << np.arange(6, dtype=np.int64)
    pattern_code = (availability.astype(np.int64) * weights).sum(axis=1)
    codes, counts = np.unique(pattern_code, return_counts=True)
    require_exact(payload[key("availability_pattern_code")], pattern_code,
                  f"{label} availability-pattern code is inconsistent")
    require_exact(payload[key("availability_pattern_codes")], codes,
                  f"{label} availability-pattern code list is inconsistent")
    require_exact(payload[key("availability_pattern_counts")], counts,
                  f"{label} availability-pattern counts are inconsistent")
    require_exact(codes, np.asarray([55, 63]), f"{label} availability patterns drifted")
    require_exact(counts, np.asarray([243, 2191]), f"{label} availability counts drifted")
    first = np.asarray([calendar[pattern_code == code].min() for code in codes])
    last = np.asarray([calendar[pattern_code == code].max() for code in codes])
    require_exact(payload[key("availability_pattern_first_date_ns")], first,
                  f"{label} first dates by availability pattern disagree")
    require_exact(payload[key("availability_pattern_last_date_ns")], last,
                  f"{label} last dates by availability pattern disagree")
    R_z = np.asarray(payload[key("R_z")], dtype=float)
    L_z = np.asarray(payload[key("L_z")], dtype=float)
    require(R_z.shape == L_z.shape == (6, 6), f"{label} dependence matrices have wrong shape")
    require(np.all(np.isfinite(R_z)) and np.all(np.isfinite(L_z)),
            f"{label} dependence matrices contain non-finite values")
    require(np.allclose(R_z, R_z.T, rtol=0.0, atol=1e-12), f"{label} R_z is not symmetric")
    require(np.allclose(np.diag(R_z), 1.0, rtol=0.0, atol=1e-12),
            f"{label} R_z diagonal is not one")
    require(np.max(np.abs(R_z)) <= 1.0 + 1e-12,
            f"{label} R_z contains values outside the correlation range")
    require(np.allclose(L_z, np.tril(L_z), rtol=0.0, atol=1e-15) and
            np.all(np.diag(L_z) > 0), f"{label} L_z is not a valid lower Cholesky factor")
    # The producer applies a common diagonal jitter only if R_z itself is not PD.
    # Validate that documented fallback instead of incorrectly requiring L L' == R_z.
    gram_delta = L_z @ L_z.T - R_z
    off_diagonal = gram_delta.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    diagonal_shift = np.diag(gram_delta)
    require(np.allclose(off_diagonal, 0.0, rtol=0.0, atol=1e-12),
            f"{label} L_z L_z' differs from R_z outside the diagonal")
    require(np.allclose(diagonal_shift, diagonal_shift[0], rtol=0.0, atol=1e-12),
            f"{label} Cholesky adjustment is not one common diagonal jitter")
    jitter = float(diagonal_shift.mean())
    require(jitter >= -1e-12,
            f"{label} Cholesky factor implies a negative diagonal adjustment")
    if jitter > 1e-12:
        ladder_step = int(round(math.log10(jitter / 1e-8)))
        require(ladder_step >= 0, f"{label} Cholesky jitter is below the producer's first step")
        expected_jitter = 1e-8 * (10.0 ** ladder_step)
        require(math.isclose(jitter, expected_jitter, rel_tol=1e-6, abs_tol=1e-12),
                f"{label} Cholesky jitter is not on the producer's power-of-ten ladder")
        previous_jitter = 0.0 if ladder_step == 0 else expected_jitter / 10.0
        try:
            np.linalg.cholesky(R_z + previous_jitter * np.eye(6))
        except np.linalg.LinAlgError:
            pass
        else:
            raise AssertionError(f"{label} Cholesky jitter is larger than necessary")
    return {
        "calendar": calendar,
        "positions": positions,
        "offsets": offsets,
        "availability": availability,
        "codes": codes,
        "counts": counts,
    }


def _validate_c9_audit(payload, spec, label):
    required = {
        "audit_spec", "audit_design_returns_concat", "audit_design_row_offsets",
        "audit_design_index_ns_concat", "audit_design_index_offsets",
        "audit_design_exog_unr_concat", "audit_design_exog_unr_offsets",
        "audit_design_exog_null_concat", "audit_design_exog_null_offsets",
        "audit_null_params_by_asset", "audit_null_mean_returns",
        "audit_null_sigma2_concat", "audit_null_sigma2_offsets",
        "audit_common_pos_concat", "audit_common_pos_offsets",
        "audit_nu_null_by_asset", "audit_nu_c_null", "audit_use_t_copula",
        "audit_refitter_contract", "audit_simulator_contract", *HASH_KEYS,
    }
    require_fields(payload, required, label)
    require(scalar_text(payload, "audit_spec") == spec, f"{label} audit spec is mislabeled")
    calendar = _validate_union_calendar(payload, "audit_", label)
    require(bool(scalar(payload, "audit_use_t_copula")), f"{label} is not a t-copula audit")
    nu = np.asarray(payload["audit_nu_null_by_asset"], dtype=float)
    require(nu.shape == (6,) and np.all(nu > 2), f"{label} fitted marginal nu vector is invalid")
    require_near(scalar(payload, "audit_nu_c_null"), np.median(nu),
                 f"{label} copula df is not the median fitted marginal df")

    row_offsets = np.asarray(payload["audit_design_row_offsets"], dtype=np.int64)
    index_offsets = np.asarray(payload["audit_design_index_offsets"], dtype=np.int64)
    unr_offsets = np.asarray(payload["audit_design_exog_unr_offsets"], dtype=np.int64)
    null_offsets = np.asarray(payload["audit_design_exog_null_offsets"], dtype=np.int64)
    sigma_offsets = np.asarray(payload["audit_null_sigma2_offsets"], dtype=np.int64)
    for name, offsets in (("row", row_offsets), ("index", index_offsets),
                          ("unrestricted-exog", unr_offsets),
                          ("null-exog", null_offsets), ("sigma2", sigma_offsets)):
        require(offsets.shape == (7,) and offsets[0] == 0 and np.all(np.diff(offsets) >= 0),
                f"{label} {name} offsets are malformed")
    require_exact(np.diff(row_offsets), np.diff(index_offsets),
                  f"{label} return/index per-asset lengths disagree")
    require_exact(np.diff(row_offsets), np.diff(unr_offsets),
                  f"{label} return/unrestricted-exog lengths disagree")
    require_exact(np.diff(row_offsets), np.diff(null_offsets),
                  f"{label} return/null-exog lengths disagree")
    require_exact(np.diff(row_offsets), np.diff(sigma_offsets),
                  f"{label} return/null-path lengths disagree")
    require(row_offsets[-1] == len(payload["audit_design_returns_concat"]),
            f"{label} return offsets do not close")
    require(index_offsets[-1] == len(payload["audit_design_index_ns_concat"]),
            f"{label} index offsets do not close")
    require(unr_offsets[-1] == len(payload["audit_design_exog_unr_concat"]),
            f"{label} unrestricted-exog offsets do not close")
    require(null_offsets[-1] == len(payload["audit_design_exog_null_concat"]),
            f"{label} null-exog offsets do not close")
    require(sigma_offsets[-1] == len(payload["audit_null_sigma2_concat"]),
            f"{label} null-path offsets do not close")
    require(np.asarray(payload["audit_design_exog_unr_concat"]).ndim == 2,
            f"{label} unrestricted exogenous design is not a matrix")
    require(np.asarray(payload["audit_design_exog_null_concat"]).ndim == 2,
            f"{label} null exogenous design is not a matrix")
    index_ns = np.asarray(payload["audit_design_index_ns_concat"], dtype=np.int64)
    for j, asset in enumerate(ASSETS):
        archived_index = index_ns[index_offsets[j]:index_offsets[j + 1]]
        pos = calendar["positions"][calendar["offsets"][j]:calendar["offsets"][j + 1]]
        require_exact(archived_index, calendar["calendar"][pos],
                      f"{label} {asset} archived index/calendar mapping disagrees")

    design_named = [
        ("asset_names", payload["audit_asset_names"]),
        ("returns_concat", payload["audit_design_returns_concat"]),
        ("row_offsets", payload["audit_design_row_offsets"]),
        ("index_ns_concat", payload["audit_design_index_ns_concat"]),
        ("index_offsets", payload["audit_design_index_offsets"]),
        ("exog_unr_concat", payload["audit_design_exog_unr_concat"]),
        ("exog_unr_offsets", payload["audit_design_exog_unr_offsets"]),
        ("exog_null_concat", payload["audit_design_exog_null_concat"]),
        ("exog_null_offsets", payload["audit_design_exog_null_offsets"]),
        ("simulation_calendar_ns", payload["audit_simulation_calendar_ns"]),
        ("calendar_pos_concat", payload["audit_calendar_pos_concat"]),
        ("calendar_pos_offsets", payload["audit_calendar_pos_offsets"]),
        ("availability_matrix", payload["audit_availability_matrix"]),
    ]
    dgp_named = [
        ("asset_names", payload["audit_asset_names"]),
        ("null_params", payload["audit_null_params_by_asset"]),
        ("mean_returns", payload["audit_null_mean_returns"]),
        ("sigma2_null_concat", payload["audit_null_sigma2_concat"]),
        ("sigma2_null_offsets", payload["audit_null_sigma2_offsets"]),
        ("common_pos_concat", payload["audit_common_pos_concat"]),
        ("common_pos_offsets", payload["audit_common_pos_offsets"]),
        ("simulation_calendar_ns", payload["audit_simulation_calendar_ns"]),
        ("calendar_pos_concat", payload["audit_calendar_pos_concat"]),
        ("calendar_pos_offsets", payload["audit_calendar_pos_offsets"]),
        ("nu_null", payload["audit_nu_null_by_asset"]),
        ("nu_c_null", payload["audit_nu_c_null"]),
        ("R_z", payload["audit_R_z"]),
        ("L_z", payload["audit_L_z"]),
        ("use_t_copula", payload["audit_use_t_copula"]),
    ]
    design_hash = hash_named_arrays(design_named)
    dgp_hash = hash_named_arrays(dgp_named)
    refitter_contract = scalar_text(payload, "audit_refitter_contract")
    simulator_contract = scalar_text(payload, "audit_simulator_contract")
    current_refitter, current_simulator = _current_c9_contracts()
    require(refitter_contract == current_refitter,
            f"{label} refitter contract does not match the current packaged source")
    require(simulator_contract == current_simulator,
            f"{label} simulator contract does not match the current packaged source")
    refitter_hash = hashlib.sha256(refitter_contract.encode()).hexdigest()
    simulator_hash = hashlib.sha256(simulator_contract.encode()).hexdigest()
    expected_hashes = {
        "audit_design_sha256": design_hash,
        "audit_fixed_null_dgp_sha256": dgp_hash,
        "audit_refitter_sha256": refitter_hash,
        "audit_simulator_sha256": simulator_hash,
    }
    analysis_hash = hashlib.sha256(
        (spec + "\n" + design_hash + "\n" + dgp_hash + "\n"
         + refitter_hash + "\n" + simulator_hash).encode()
    ).hexdigest()
    expected_hashes["audit_analysis_stack_sha256"] = analysis_hash
    for key, expected in expected_hashes.items():
        actual = require_hash(scalar_text(payload, key), f"{label} {key}")
        require(actual == expected, f"{label} {key} does not fingerprint its archived payload")
    return expected_hashes


def validate_c9():
    csv_path = RESULTS / "c9-tcopula-results.csv"
    require(csv_path.is_file(), "c9 result CSV is missing")
    frame = pd.read_csv(csv_path, float_precision="round_trip")
    required_csv = {
        "spec", "base_seed", "draw_seed_offset", "first_draw_seed",
        "last_draw_seed", "B_requested", "n_calendar",
        "availability_pattern_codes", "availability_pattern_counts",
        "design_sha256", "fixed_null_dgp_sha256", "refitter_sha256",
        "simulator_sha256", "analysis_stack_sha256", "multiplier", "d_bar_obs",
        "rho_return", "rho_resid",
        "nu_null_median", "p_tcopula_one_sided", "p_tcopula_two_sided",
        "null_sd_tcopula", "B_used_t", "n_dropped_t", "frac_dropped_t",
        "n_upper_tail_hits_t", "n_abs_stat_hits_t", "n_lower_tail_hits_t",
        "n_draws_with_six_start_rescue_attempt_t",
        "n_draws_retained_after_six_start_rescue_t",
        "n_draws_failed_after_six_start_rescue_t",
        "n_asset_refits_with_six_start_rescue_attempt_t",
        "n_asset_refits_rescued_by_six_start_t", "gaussian_validation_run",
        "p_gaussian_old_one_sided", "null_sd_gaussian", "B_used_gaussian",
        "n_dropped_gaussian", "frac_dropped_gaussian",
        "n_upper_tail_hits_gaussian", "sd_widened",
    }
    require_fields(frame.columns, required_csv, "c9 results CSV")
    rows = indexed_rows(frame, "spec", SPECS, "c9 results CSV")
    expected_seeds = np.arange(
        FIT_SEED + DRAW_SEED_OFFSET,
        FIT_SEED + DRAW_SEED_OFFSET + 2000,
        dtype=np.int64,
    )
    payloads = {}
    common_draw_seeds = None
    for spec in SPECS:
        label = f"c9 {spec}"
        payload = load_npz(RESULTS / f"c9-tcopula-draws-{spec}.npz")
        required = {
            "null_t", "null_gaussian", "null_t_full", "null_t_usable_mask",
            "null_t_excluded_mask", "null_gaussian_full",
            "null_gaussian_usable_mask", "null_gaussian_excluded_mask",
            "gaussian_validation_run", "draw_seeds", "seed_schema_version",
            "seed_schema", "base_seed", "draw_seed_offset", "first_draw_seed",
            "last_draw_seed", "B_requested", "d_bar_obs",
            "draw_had_six_start_rescue_attempt_t",
            "draw_retained_after_six_start_rescue_t",
            "n_asset_refits_with_six_start_rescue_attempt_t",
            "n_asset_refits_rescued_by_six_start_t",
        }
        require_fields(payload, required, label)
        require(int(scalar(payload, "seed_schema_version")) == 2,
                f"{label} does not use seed schema v2")
        require(int(scalar(payload, "base_seed")) == FIT_SEED,
                f"{label} fit/base seed drifted")
        require(int(scalar(payload, "draw_seed_offset")) == DRAW_SEED_OFFSET,
                f"{label} draw-seed offset drifted")
        require(int(scalar(payload, "B_requested")) == 2000,
                f"{label} must retain 2,000 requested draw positions")
        seeds = np.asarray(payload["draw_seeds"], dtype=np.int64)
        require_exact(seeds, expected_seeds, f"{label} draw-seed array drifted")
        require((int(scalar(payload, "first_draw_seed")),
                 int(scalar(payload, "last_draw_seed"))) ==
                (int(expected_seeds[0]), int(expected_seeds[-1])),
                f"{label} first/last seed fields disagree with draw_seeds")
        if common_draw_seeds is None:
            common_draw_seeds = seeds
        else:
            require_exact(seeds, common_draw_seeds,
                          f"{label} is not paired to the other c9 specifications")

        full_t = np.asarray(payload["null_t_full"], dtype=float)
        usable_t = np.asarray(payload["null_t_usable_mask"], dtype=bool)
        excluded_t = np.asarray(payload["null_t_excluded_mask"], dtype=bool)
        require(full_t.shape == usable_t.shape == excluded_t.shape == (2000,),
                f"{label} Student-t draw/mask shapes drifted")
        require(not np.any(np.isinf(full_t)),
                f"{label} Student-t full draw vector contains infinities")
        require_exact(usable_t, ~np.isnan(full_t), f"{label} Student-t usable mask disagrees")
        require_exact(excluded_t, ~usable_t, f"{label} Student-t masks are not complements")
        require_exact(payload["null_t"], full_t[usable_t],
                      f"{label} compressed Student-t draws disagree with full vector")

        full_g = np.asarray(payload["null_gaussian_full"], dtype=float)
        usable_g = np.asarray(payload["null_gaussian_usable_mask"], dtype=bool)
        excluded_g = np.asarray(payload["null_gaussian_excluded_mask"], dtype=bool)
        require(bool(scalar(payload, "gaussian_validation_run")),
                f"{label} final archive skipped the Gaussian comparator")
        require(full_g.shape == usable_g.shape == excluded_g.shape == (2000,),
                f"{label} Gaussian draw/mask shapes drifted")
        require(not np.any(np.isinf(full_g)),
                f"{label} Gaussian full draw vector contains infinities")
        require_exact(usable_g, ~np.isnan(full_g), f"{label} Gaussian usable mask disagrees")
        require_exact(excluded_g, ~usable_g, f"{label} Gaussian masks are not complements")
        require_exact(payload["null_gaussian"], full_g[usable_g],
                      f"{label} compressed Gaussian draws disagree with full vector")

        attempted = np.asarray(payload["draw_had_six_start_rescue_attempt_t"], dtype=bool)
        retained = np.asarray(payload["draw_retained_after_six_start_rescue_t"], dtype=bool)
        asset_attempts = np.asarray(
            payload["n_asset_refits_with_six_start_rescue_attempt_t"], dtype=np.int64
        )
        asset_successes = np.asarray(
            payload["n_asset_refits_rescued_by_six_start_t"], dtype=np.int64
        )
        for name, array in (("rescue attempts", attempted), ("rescue retention", retained),
                            ("asset rescue attempts", asset_attempts),
                            ("asset rescue successes", asset_successes)):
            require(array.shape == (2000,), f"{label} {name} has wrong shape")
        require(np.all(asset_attempts >= 0) and np.all(asset_successes >= 0),
                f"{label} rescue telemetry contains negative counts")
        require(np.all(asset_successes <= asset_attempts),
                f"{label} rescued-asset counts exceed attempts")
        require_exact(attempted, asset_attempts > 0,
                      f"{label} draw rescue flags disagree with asset attempts")
        require_exact(retained, attempted & usable_t,
                      f"{label} rescue-retention flags disagree with usable draws")

        hashes = _validate_c9_audit(payload, spec, label)
        row = rows.loc[spec]
        for field, expected in (
            ("base_seed", FIT_SEED), ("draw_seed_offset", DRAW_SEED_OFFSET),
            ("first_draw_seed", expected_seeds[0]),
            ("last_draw_seed", expected_seeds[-1]), ("B_requested", 2000),
            ("n_calendar", 2434), ("B_used_t", usable_t.sum()),
            ("n_dropped_t", excluded_t.sum()), ("B_used_gaussian", usable_g.sum()),
            ("n_dropped_gaussian", excluded_g.sum()),
        ):
            _csv_int(row, field, expected, label)
        require(str(row["availability_pattern_codes"]) == "55;63" and
                str(row["availability_pattern_counts"]) == "243;2191",
                f"{label} CSV availability summary drifted")
        hash_columns = {
            "design_sha256": "audit_design_sha256",
            "fixed_null_dgp_sha256": "audit_fixed_null_dgp_sha256",
            "refitter_sha256": "audit_refitter_sha256",
            "simulator_sha256": "audit_simulator_sha256",
            "analysis_stack_sha256": "audit_analysis_stack_sha256",
        }
        for csv_key, audit_key in hash_columns.items():
            require(str(row[csv_key]) == hashes[audit_key],
                    f"{label} CSV {csv_key} differs from its NPZ audit")
        values_t = full_t[usable_t]
        values_g = full_g[usable_g]
        d_obs = float(scalar(payload, "d_bar_obs"))
        R_z = np.asarray(payload["audit_R_z"], dtype=float)
        rho_resid = (R_z.sum() - np.trace(R_z)) / 30.0
        returns_matrix = np.full((2434, 6), np.nan, dtype=float)
        returns_concat = np.asarray(payload["audit_design_returns_concat"], dtype=float)
        row_offsets = np.asarray(payload["audit_design_row_offsets"], dtype=np.int64)
        calendar_positions = np.asarray(payload["audit_calendar_pos_concat"], dtype=np.int64)
        calendar_offsets = np.asarray(payload["audit_calendar_pos_offsets"], dtype=np.int64)
        for j in range(6):
            returns_matrix[
                calendar_positions[calendar_offsets[j]:calendar_offsets[j + 1]], j
            ] = returns_concat[row_offsets[j]:row_offsets[j + 1]]
        complete_returns = returns_matrix[np.all(np.isfinite(returns_matrix), axis=1)]
        R_return = np.corrcoef(complete_returns, rowvar=False)
        rho_return = (R_return.sum() - np.trace(R_return)) / 30.0
        upper_t = int(np.sum(values_t >= d_obs))
        absolute_t = int(np.sum(np.abs(values_t) >= abs(d_obs)))
        lower_t = int(np.sum(values_t <= d_obs))
        upper_g = int(np.sum(values_g >= d_obs))
        for field, expected in (
            ("n_upper_tail_hits_t", upper_t), ("n_abs_stat_hits_t", absolute_t),
            ("n_lower_tail_hits_t", lower_t),
            ("n_draws_with_six_start_rescue_attempt_t", attempted.sum()),
            ("n_draws_retained_after_six_start_rescue_t", retained.sum()),
            ("n_draws_failed_after_six_start_rescue_t", np.sum(attempted & excluded_t)),
            ("n_asset_refits_with_six_start_rescue_attempt_t", asset_attempts.sum()),
            ("n_asset_refits_rescued_by_six_start_t", asset_successes.sum()),
            ("n_upper_tail_hits_gaussian", upper_g),
        ):
            _csv_int(row, field, expected, label)
        for field, expected in (
            ("d_bar_obs", d_obs),
            ("rho_return", rho_return),
            ("rho_resid", rho_resid),
            ("nu_null_median", np.median(payload["audit_nu_null_by_asset"])),
            ("p_tcopula_one_sided", smoothed_tail(upper_t, len(values_t))),
            ("p_tcopula_two_sided", smoothed_tail(absolute_t, len(values_t))),
            ("null_sd_tcopula", values_t.std()),
            ("frac_dropped_t", excluded_t.mean()),
            ("p_gaussian_old_one_sided", smoothed_tail(upper_g, len(values_g))),
            ("null_sd_gaussian", values_g.std()),
            ("frac_dropped_gaussian", excluded_g.mean()),
        ):
            _csv_float(row, field, expected, label)
        require(np.isfinite(float(row["multiplier"])) and float(row["multiplier"]) > 0 and
                not math.isclose(float(row["multiplier"]), 1.0, rel_tol=0.0, abs_tol=1e-12),
                f"{label} CSV multiplier is invalid")
        require(bool(row["gaussian_validation_run"]),
                f"{label} CSV does not record the Gaussian comparator")
        require(bool(row["sd_widened"]) == bool(values_t.std() > values_g.std()),
                f"{label} CSV sd_widened flag is inconsistent")
        payloads[spec] = payload
    finding_snippets = []
    for spec in SPECS:
        row = rows.loc[spec]
        finding_snippets.extend([
            (
                f"{spec} result row",
                f"| {spec} | {float(row['multiplier']):.2f}x | "
                f"{float(row['p_gaussian_old_one_sided']):.4f} | "
                f"**{float(row['p_tcopula_one_sided']):.4f}** | "
                f"{float(row['p_tcopula_two_sided']):.4f} | "
                f"{float(row['null_sd_gaussian']):.4f} -> "
                f"{float(row['null_sd_tcopula']):.4f} |",
            ),
            (
                f"{spec} retention row",
                f"| {spec} | {int(row['B_used_t'])} / {int(row['B_requested'])} | "
                f"{int(row['n_dropped_t'])} | "
                f"{int(row['n_draws_with_six_start_rescue_attempt_t'])} / "
                f"{int(row['n_draws_retained_after_six_start_rescue_t'])} / "
                f"{int(row['n_draws_failed_after_six_start_rescue_t'])} | "
                f"{int(row['n_asset_refits_with_six_start_rescue_attempt_t'])} / "
                f"{int(row['n_asset_refits_rescued_by_six_start_t'])} | "
                f"{int(row['B_used_gaussian'])} / {int(row['B_requested'])} | "
                f"{int(row['n_dropped_gaussian'])} |",
            ),
        ])
    require_finding(
        RESULTS / "c9-tcopula-bootstrap-FINDING.md",
        "# C9 -- Student-t-copula CCC-GARCH-X fixed-path bootstrap",
        finding_snippets,
        "c9",
    )
    return rows, payloads


C8H_SOURCE_SHA256 = "ea50efe3a34c14cdcb0b76dcb80ba9dea3c9fa32bec2c92da7cd5531cb3aaf6f"
C9_CURRENT_SOURCE_SHA256 = "7bfe802269fd85104ea8e68f40b19c04dcdf1577f0ba562c54ab57df1fd37aa1"
C20_C9_SOURCE_AT_RUN_SHA256 = "b5a3c78ce31497ca453495c550fc76b061e5b0cc4da5ded57aa035e93923e3c8"


def validate_c8h(c9_rows, c9_payloads):
    """Bind the intermediate C8h comparator to C9's audited Gaussian legs."""
    label = "c8h"
    csv_path = RESULTS / "c8h-break-ccc-bootstrap-results.csv"
    source_path = HERE / "c8h_break_controls_ccc_bootstrap.py"
    require(csv_path.is_file(), "c8h result CSV is missing")
    require(sha256_file(source_path) == C8H_SOURCE_SHA256,
            "c8h source differs from the release-audited implementation")
    frame = pd.read_csv(csv_path, float_precision="round_trip")
    required_csv = {
        "variant", "multiplier", "d_bar_obs", "mean_infra", "mean_reg",
        "rho_return", "rho_resid", "n_calendar",
        "availability_pattern_codes", "availability_pattern_counts",
        "naive_welch_t", "naive_welch_p", "robust_p_one_sided",
        "robust_p_two_sided", "null_dbar_mean", "null_dbar_sd", "B_used",
        "frac_dropped", "per_asset_diff",
    }
    require_fields(frame.columns, required_csv, "c8h results CSV")
    variants = ("full", "crisis")
    rows = indexed_rows(frame, "variant", variants, "c8h results CSV")
    expected_naive = {
        "full": (2.5219294334774034, 0.03029562498501582),
        "crisis": (3.0744679713090024, 0.012429420051725593),
    }
    finding_snippets = []
    for variant in variants:
        display_variant = "asset-specific high-variance" if variant == "crisis" else variant
        row = rows.loc[variant]
        payload = load_npz(RESULTS / f"c8h-break-ccc-draws-{variant}.npz")
        require_fields(
            payload,
            {
                "null", "d_bar_obs", "simulation_calendar_ns",
                "availability_pattern_codes", "availability_pattern_counts",
            },
            f"c8h {variant}",
        )
        parent = c9_payloads[variant]
        values = np.asarray(payload["null"], dtype=float)
        require_exact(values, parent["null_gaussian"],
                      f"c8h {variant} draws differ from C9's Gaussian leg")
        require_exact(payload["d_bar_obs"], parent["d_bar_obs"],
                      f"c8h {variant} observed statistic differs from C9")
        for local_key, parent_key in (
            ("simulation_calendar_ns", "audit_simulation_calendar_ns"),
            ("availability_pattern_codes", "audit_availability_pattern_codes"),
            ("availability_pattern_counts", "audit_availability_pattern_counts"),
        ):
            require_exact(payload[local_key], parent[parent_key],
                          f"c8h {variant} {local_key} differs from C9")
        d_obs = float(scalar(payload, "d_bar_obs"))
        upper = int(np.sum(values >= d_obs))
        absolute = int(np.sum(np.abs(values) >= abs(d_obs)))
        _csv_int(row, "B_used", len(values), f"c8h {variant}")
        _csv_int(row, "n_calendar", 2434, f"c8h {variant}")
        require(
            str(row["availability_pattern_codes"]) == "55;63"
            and str(row["availability_pattern_counts"]) == "243;2191",
            f"c8h {variant} CSV availability summary drifted",
        )
        for field, expected in (
            ("d_bar_obs", d_obs),
            ("multiplier", c9_rows.loc[variant, "multiplier"]),
            ("rho_return", c9_rows.loc[variant, "rho_return"]),
            ("rho_resid", c9_rows.loc[variant, "rho_resid"]),
            ("robust_p_one_sided", smoothed_tail(upper, len(values))),
            ("robust_p_two_sided", smoothed_tail(absolute, len(values))),
            ("null_dbar_mean", values.mean()),
            ("null_dbar_sd", values.std()),
            ("frac_dropped", (2000 - len(values)) / 2000),
            ("naive_welch_t", expected_naive[variant][0]),
            ("naive_welch_p", expected_naive[variant][1]),
        ):
            _csv_float(row, field, expected, f"c8h {variant}")
        require_near(
            float(row["mean_infra"]) - float(row["mean_reg"]),
            d_obs,
            f"c8h {variant} mean difference is inconsistent",
        )
        require_near(
            float(row["mean_infra"]) / float(row["mean_reg"]),
            float(row["multiplier"]),
            f"c8h {variant} multiplier is inconsistent",
        )
        per_asset = np.asarray(json.loads(str(row["per_asset_diff"])), dtype=float)
        require(per_asset.shape == (6,) and np.all(np.isfinite(per_asset)),
                f"c8h {variant} per-asset difference vector is malformed")
        require_near(per_asset.mean(), d_obs,
                     f"c8h {variant} per-asset differences do not average to d_bar_obs")
        finding_snippets.extend([
            (
                f"{variant} result row",
                f"| {display_variant} regime | {float(row['multiplier']):.2f}x | "
                f"{float(row['naive_welch_p']):.4f} | "
                f"**{float(row['robust_p_one_sided']):.4f}** | "
                f"{float(row['robust_p_two_sided']):.4f} | "
                f"{float(row['rho_resid']):.3f} | "
                f"{float(row['frac_dropped']):.1%} |",
            ),
            (
                f"{variant} paired archive hashes",
                f"- {display_variant.capitalize()} C8h/C9 draw-archive SHA-256: "
                f"`{sha256_file(RESULTS / f'c8h-break-ccc-draws-{variant}.npz')}` / "
                f"`{sha256_file(RESULTS / f'c9-tcopula-draws-{variant}.npz')}`",
            ),
        ])
    finding_snippets.extend([
        ("C8h source hash", f"- C8h source SHA-256: `{C8H_SOURCE_SHA256}`"),
        (
            "C8h CSV hash",
            f"- C8h results CSV SHA-256: `{sha256_file(csv_path)}`",
        ),
        (
            "legacy absolute-statistic field",
            "The legacy CSV field `robust_p_two_sided` stores the unrecentred "
            "absolute-statistic tail",
        ),
    ])
    require_finding(
        RESULTS / "c8h-break-controls-ccc-FINDING.md",
        "# C8h — Intermediate Gaussian-Innovation Break-Control Bootstrap",
        finding_snippets,
        label,
    )


C9_TO_PORTABLE_DGP = {
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
    "availability_pattern_first_date_ns": "audit_availability_pattern_first_date_ns",
    "availability_pattern_last_date_ns": "audit_availability_pattern_last_date_ns",
}


def _require_baseline_dgp_identity(payload, c9_baseline, label):
    require_fields(payload, C9_TO_PORTABLE_DGP, label)
    for local_key, c9_key in C9_TO_PORTABLE_DGP.items():
        require_exact(payload[local_key], c9_baseline[c9_key],
                      f"{label} {local_key} differs exactly from c9 baseline")
    for key in HASH_KEYS:
        require_fields(payload, {key}, label)
        require(scalar_text(payload, key) == scalar_text(c9_baseline, key),
                f"{label} {key} differs from c9 baseline")


def validate_c10(c9_baseline, c9_baseline_row):
    label = "c10"
    draws_path = RESULTS / "c10-size-study-draws.npz"
    results_path = RESULTS / "c10-size-study-results.csv"
    metadata_path = RESULTS / "c10-size-study-metadata.csv"
    require(results_path.is_file(), "required c10 results CSV is missing")
    require(metadata_path.is_file(), "required c10 metadata CSV is missing")
    draws = load_npz(draws_path)
    required = {
        "dbar_panels", "p_iid", "p_de", "g_draws", "t_draws",
        "crit_gauss", "crit_t", "panel_seeds", "panel_usable_mask",
        "panel_excluded_mask", "panel_se_ok_full", "panel_delta_infra_full",
        "panel_delta_reg_full", "panel_se_infra_full", "panel_se_reg_full",
        "panel_dbar_full", "panel_p_iid_full", "panel_p_de_full",
        "gaussian_reference_seeds", "gaussian_reference_full",
        "gaussian_reference_usable_mask", "gaussian_reference_excluded_mask",
        "t_reference_seeds", "t_reference_full", "t_reference_usable_mask",
        "t_reference_excluded_mask", "c10_seed_schema_version", "fit_seed",
        "mc_seed", "seed", "rescue_seed", "panel_first_seed", "panel_last_seed",
        "gaussian_reference_first_seed", "gaussian_reference_last_seed",
        "t_reference_first_seed", "t_reference_last_seed", "reference_seed_offset",
        "t_reference_seed_offset", "N_requested", "B_ref_requested",
        "c9_seed_schema_version", "c9_exact_identity_verified",
        "c10_analysis_contract_version", "c10_analysis_contract",
        "c10_analysis_sha256", "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    }
    require_fields(draws, required, label)
    require(int(scalar(draws, "c10_seed_schema_version")) == 1,
            "c10 seed schema version drifted")
    require(int(scalar(draws, "c9_seed_schema_version")) == 2 and
            bool(scalar(draws, "c9_exact_identity_verified")),
            "c10 lacks its exact c9 schema-v2 identity certification")
    require((int(scalar(draws, "fit_seed")), int(scalar(draws, "mc_seed")),
             int(scalar(draws, "rescue_seed"))) == (FIT_SEED, MC_SEED, RESCUE_SEED),
            "c10 fit/Monte-Carlo/rescue seed contract drifted")
    require(int(scalar(draws, "seed")) == MC_SEED, "c10 legacy seed alias is inconsistent")
    require((int(scalar(draws, "reference_seed_offset")),
             int(scalar(draws, "t_reference_seed_offset"))) == (700_000, 500_000),
            "c10 reference seed offsets drifted")
    N = int(scalar(draws, "N_requested"))
    B_ref = int(scalar(draws, "B_ref_requested"))
    require((N, B_ref) == (1000, 4000),
            "c10 final release must use N=1,000 and B_ref=4,000")
    panel_seeds = np.asarray(draws["panel_seeds"], dtype=np.int64)
    g_seeds = np.asarray(draws["gaussian_reference_seeds"], dtype=np.int64)
    t_seeds = np.asarray(draws["t_reference_seeds"], dtype=np.int64)
    expected_panel = np.arange(MC_SEED, MC_SEED + N, dtype=np.int64)
    expected_g = np.arange(MC_SEED + 700_000, MC_SEED + 700_000 + B_ref, dtype=np.int64)
    expected_t = expected_g + 500_000
    require_exact(panel_seeds, expected_panel, "c10 outer-panel seed array drifted")
    require_exact(g_seeds, expected_g, "c10 Gaussian-reference seed array drifted")
    require_exact(t_seeds, expected_t, "c10 Student-t-reference seed array drifted")
    for field, expected in (
        ("panel_first_seed", expected_panel[0]), ("panel_last_seed", expected_panel[-1]),
        ("gaussian_reference_first_seed", expected_g[0]),
        ("gaussian_reference_last_seed", expected_g[-1]),
        ("t_reference_first_seed", expected_t[0]),
        ("t_reference_last_seed", expected_t[-1]),
    ):
        require(int(scalar(draws, field)) == int(expected), f"c10 {field} is inconsistent")

    panel_mask = np.asarray(draws["panel_usable_mask"], dtype=bool)
    panel_excluded = np.asarray(draws["panel_excluded_mask"], dtype=bool)
    dbar = np.asarray(draws["panel_dbar_full"], dtype=float)
    p_iid = np.asarray(draws["panel_p_iid_full"], dtype=float)
    p_de = np.asarray(draws["panel_p_de_full"], dtype=float)
    di = np.asarray(draws["panel_delta_infra_full"], dtype=float)
    dr = np.asarray(draws["panel_delta_reg_full"], dtype=float)
    se_i = np.asarray(draws["panel_se_infra_full"], dtype=float)
    se_r = np.asarray(draws["panel_se_reg_full"], dtype=float)
    se_ok = np.asarray(draws["panel_se_ok_full"], dtype=bool)
    require(panel_mask.shape == panel_excluded.shape == dbar.shape == p_iid.shape ==
            p_de.shape == se_ok.shape == (N,), "c10 panel-level array shapes drifted")
    require(di.shape == dr.shape == se_i.shape == se_r.shape == (N, 6),
            "c10 per-asset panel array shapes drifted")
    require_exact(panel_excluded, ~panel_mask, "c10 panel masks are not complements")
    require_exact(panel_mask, ~np.isnan(dbar), "c10 panel mask does not identify dbar NaNs")
    require(np.all(np.isfinite(di[panel_mask])) and np.all(np.isfinite(dr[panel_mask])),
            "c10 usable panels contain non-finite coefficient values")
    require(np.all(np.isnan(di[~panel_mask])) and np.all(np.isnan(dr[~panel_mask])) and
            np.all(np.isnan(se_i[~panel_mask])) and np.all(np.isnan(se_r[~panel_mask])) and
            np.all(np.isnan(p_iid[~panel_mask])) and np.all(np.isnan(p_de[~panel_mask])),
            "c10 excluded panels do not use NaN consistently")
    expected_se_ok = panel_mask & np.all(np.isfinite(se_i) & np.isfinite(se_r), axis=1)
    require_exact(se_ok, expected_se_ok,
                  "c10 panel SE flags do not identify six finite SE pairs")
    require(not np.any(np.isinf(se_i)) and not np.any(np.isinf(se_r)) and
            not np.any(np.isinf(p_iid)) and not np.any(np.isinf(p_de)),
            "c10 SE or p-value arrays contain infinities")
    require(np.allclose(dbar[panel_mask], np.mean(di[panel_mask] - dr[panel_mask], axis=1),
                        rtol=0.0, atol=1e-12),
            "c10 panel dbar does not equal the mean per-asset difference")
    for name, values in (("iid p", p_iid), ("design-effect p", p_de)):
        finite = values[np.isfinite(values)]
        require(np.all((0 <= finite) & (finite <= 1)), f"c10 {name} values leave [0,1]")
    require_exact(draws["dbar_panels"], dbar[panel_mask],
                  "c10 compressed/full dbar arrays disagree")
    require_exact(draws["p_iid"], p_iid[panel_mask],
                  "c10 compressed/full iid-p arrays disagree")
    require_exact(draws["p_de"], p_de[panel_mask],
                  "c10 compressed/full design-effect-p arrays disagree")

    reference = {}
    for prefix, seeds in (("gaussian", g_seeds), ("t", t_seeds)):
        full = np.asarray(draws[f"{prefix}_reference_full"], dtype=float)
        usable = np.asarray(draws[f"{prefix}_reference_usable_mask"], dtype=bool)
        excluded = np.asarray(draws[f"{prefix}_reference_excluded_mask"], dtype=bool)
        require(full.shape == usable.shape == excluded.shape == (B_ref,),
                f"c10 {prefix} reference array shapes drifted")
        require(not np.any(np.isinf(full)),
                f"c10 {prefix} reference vector contains infinities")
        require_exact(usable, ~np.isnan(full), f"c10 {prefix} reference mask disagrees")
        require_exact(excluded, ~usable, f"c10 {prefix} reference masks are not complements")
        require_exact(draws["g_draws" if prefix == "gaussian" else "t_draws"], full[usable],
                      f"c10 {prefix} compressed/full reference arrays disagree")
        reference[prefix] = (full, usable, full[usable], seeds)
    crit_g = np.asarray(draws["crit_gauss"], dtype=float)
    crit_t = np.asarray(draws["crit_t"], dtype=float)
    require(crit_g.shape == crit_t.shape == (2,), "c10 critical-value arrays have wrong shape")
    require(np.allclose(crit_g, np.percentile(reference["gaussian"][2], [95, 90]),
                        rtol=0.0, atol=1e-12),
            "c10 Gaussian critical values are not 95th/90th percentiles")
    require(np.allclose(crit_t, np.percentile(reference["t"][2], [95, 90]),
                        rtol=0.0, atol=1e-12),
            "c10 Student-t critical values are not 95th/90th percentiles")

    _validate_union_calendar(draws, "", label)
    _require_baseline_dgp_identity(draws, c9_baseline, label)
    c10_provenance = c10.current_c10_analysis_provenance(
        scalar_text(c9_baseline, "audit_analysis_stack_sha256")
    )
    require_current_provenance(draws, c10_provenance, "c10 downstream provenance")
    require(scalar_text(draws, "c10_parent_c9_analysis_stack_sha256") ==
            scalar_text(draws, "audit_analysis_stack_sha256"),
            "c10 downstream provenance is not chained to its archived c9 parent")
    results = pd.read_csv(results_path)
    methods = ("naive_iid", "design_effect", "gaussian_copula_boot", "tcopula_boot")
    required_result = {
        "method", "size_005", "se_005", "size_010", "se_010",
        "rej_005", "rej_010", "n_panels_used",
    }
    require_fields(results.columns, required_result, "c10 results CSV")
    result_rows = indexed_rows(results, "method", methods, "c10 results CSV")
    used = int(panel_mask.sum())
    decisions = {
        "naive_iid": (p_iid[panel_mask] < 0.05, p_iid[panel_mask] < 0.10),
        "design_effect": (p_de[panel_mask] < 0.05, p_de[panel_mask] < 0.10),
        "gaussian_copula_boot": (dbar[panel_mask] > crit_g[0],
                                  dbar[panel_mask] > crit_g[1]),
        "tcopula_boot": (dbar[panel_mask] > crit_t[0], dbar[panel_mask] > crit_t[1]),
    }
    for method in methods:
        row = result_rows.loc[method]
        r5, r10 = (int(mask.sum()) for mask in decisions[method])
        _csv_int(row, "n_panels_used", used, f"c10 {method}")
        _csv_int(row, "rej_005", r5, f"c10 {method}")
        _csv_int(row, "rej_010", r10, f"c10 {method}")
        for field, reject in (("size_005", r5), ("size_010", r10)):
            _csv_float(row, field, reject / used, f"c10 {method}")
        _csv_float(row, "se_005", np.sqrt((r5 / used) * (1 - r5 / used) / used),
                   f"c10 {method}")
        _csv_float(row, "se_010", np.sqrt((r10 / used) * (1 - r10 / used) / used),
                   f"c10 {method}")

    meta = pd.read_csv(metadata_path)
    require(len(meta) == 1, "c10 metadata CSV must contain exactly one row")
    required_meta = {
        "N_requested", "N_used", "panel_fail_frac", "se_pd_frac", "B_ref",
        "B_ref_used_gauss", "B_ref_used_t", "B_ref_drop_gauss", "B_ref_drop_t",
        "fit_seed", "mc_seed", "seed", "rescue_seed", "panel_first_seed",
        "panel_last_seed", "gaussian_reference_first_seed",
        "gaussian_reference_last_seed", "t_reference_first_seed",
        "t_reference_last_seed", "c9_seed_schema_version", "n_calendar",
        "availability_pattern_codes", "availability_pattern_counts", "nu_null",
        "nu_c_null", "d_bar_obs_point", "multiplier_point", "rho_return",
        "rho_resid", "crit_gauss_005", "crit_gauss_010", "crit_t_005",
        "crit_t_010", "naive_overrej_005", "naive_overrej_010", "numba",
        "design_sha256", "fixed_null_dgp_sha256", "refitter_sha256",
        "simulator_sha256", "analysis_stack_sha256", "c10_analysis_contract_version",
        "c10_analysis_sha256", "c10_source_sha256",
        "c10_parent_c9_analysis_stack_sha256",
        "c10_downstream_analysis_stack_sha256",
    }
    require_fields(meta.columns, required_meta, "c10 metadata CSV")
    meta = meta.iloc[0]
    for field, expected in (
        ("N_requested", N), ("N_used", used), ("B_ref", B_ref),
        ("B_ref_used_gauss", reference["gaussian"][1].sum()),
        ("B_ref_used_t", reference["t"][1].sum()), ("fit_seed", FIT_SEED),
        ("mc_seed", MC_SEED), ("seed", MC_SEED), ("rescue_seed", RESCUE_SEED),
        ("panel_first_seed", expected_panel[0]), ("panel_last_seed", expected_panel[-1]),
        ("gaussian_reference_first_seed", expected_g[0]),
        ("gaussian_reference_last_seed", expected_g[-1]),
        ("t_reference_first_seed", expected_t[0]),
        ("t_reference_last_seed", expected_t[-1]), ("c9_seed_schema_version", 2),
    ):
        _csv_int(meta, field, expected, "c10 metadata")
    for field, expected in (
        ("panel_fail_frac", (~panel_mask).mean()),
        ("se_pd_frac", se_ok[panel_mask].mean()),
        ("B_ref_drop_gauss", (~reference["gaussian"][1]).mean()),
        ("B_ref_drop_t", (~reference["t"][1]).mean()),
        ("crit_gauss_005", crit_g[0]), ("crit_gauss_010", crit_g[1]),
        ("crit_t_005", crit_t[0]), ("crit_t_010", crit_t[1]),
        ("naive_overrej_005", result_rows.loc["naive_iid", "size_005"] / 0.05),
        ("naive_overrej_010", result_rows.loc["naive_iid", "size_010"] / 0.10),
        ("nu_c_null", scalar(draws, "nu_c_null")),
        ("d_bar_obs_point", c9_baseline_row["d_bar_obs"]),
        ("multiplier_point", c9_baseline_row["multiplier"]),
        ("rho_return", c9_baseline_row["rho_return"]),
        ("rho_resid", c9_baseline_row["rho_resid"]),
    ):
        _csv_float(meta, field, expected, "c10 metadata")
    require(int(meta["n_calendar"]) == 2434 and
            str(meta["availability_pattern_codes"]) == "55;63" and
            str(meta["availability_pattern_counts"]) == "243;2191",
            "c10 metadata availability summary drifted")
    expected_nu_text = ";".join(f"{x:.3f}" for x in np.asarray(draws["nu_null_by_asset"]))
    require(str(meta["nu_null"]) == expected_nu_text,
            "c10 metadata marginal-nu vector differs from its archive")
    require(str(meta["numba"]).strip().lower() == str(bool(_HAVE_NUMBA)).lower(),
            "c10 metadata numba flag differs from the packaged refitter")
    for csv_key, audit_key in (
        ("design_sha256", "audit_design_sha256"),
        ("fixed_null_dgp_sha256", "audit_fixed_null_dgp_sha256"),
        ("refitter_sha256", "audit_refitter_sha256"),
        ("simulator_sha256", "audit_simulator_sha256"),
        ("analysis_stack_sha256", "audit_analysis_stack_sha256"),
    ):
        require(str(meta[csv_key]) == scalar_text(draws, audit_key),
                f"c10 metadata {csv_key} differs from its archive")
    for key, expected in c10_provenance.items():
        if key == "c10_analysis_contract":
            continue
        require(str(meta[key]) == str(expected),
                f"c10 metadata {key} differs from current-source provenance")
    finding_labels = {
        "naive_iid": "(i) NAIVE iid t-test",
        "design_effect": "(ii) design-effect corrected",
        "gaussian_copula_boot": "(iii) Gaussian-copula bootstrap",
        "tcopula_boot": "(iv) Student-t-copula bootstrap",
    }
    finding_snippets = []
    for method, display in finding_labels.items():
        row = result_rows.loc[method]
        finding_snippets.append((
            f"{method} size row",
            f"| {display} | "
            f"{'**' if method == 'naive_iid' else ''}{float(row['size_005']):.3f}"
            f"{'**' if method == 'naive_iid' else ''} +/- {float(row['se_005']):.3f} | "
            f"{'**' if method == 'naive_iid' else ''}{float(row['size_010']):.3f}"
            f"{'**' if method == 'naive_iid' else ''} +/- {float(row['se_010']):.3f} |",
        ))
    finding_snippets.append((
        "DGP dependence anchor",
        f"rho_resid = {float(meta['rho_resid']):.3f} (return-correlation rho_return = "
        f"{float(meta['rho_return']):.3f})",
    ))
    naive_row = result_rows.loc["naive_iid"]
    de_row = result_rows.loc["design_effect"]
    finding_snippets.append((
        "design-effect comparison",
        "The **design-effect t(5) rule** rejects at "
        f"{float(de_row['size_005']):.3f}/{float(de_row['size_010']):.3f}, versus "
        f"{float(naive_row['size_005']):.3f}/{float(naive_row['size_010']):.3f} "
        "for the naive rule. It therefore does not reduce over-rejection in this run.",
    ))
    finding_path = RESULTS / "c10-size-study-FINDING.md"
    require_finding(
        finding_path,
        "# C10 -- Monte-Carlo Size-Distortion Study of the Inference Ladder",
        finding_snippets,
        "c10",
    )
    finding_text = finding_path.read_text()
    require("pulls size down" not in finding_text and "it helps" not in finding_text,
            "c10 finding retains a false design-effect improvement claim")
    return result_rows, meta, draws


def validate_c10b(c9_baseline, c10_draws):
    label = "c10b"
    csv_path = RESULTS / "c10b-dgp-perturbation-results.csv"
    require(csv_path.is_file(), "required c10b results CSV is missing")
    draws = load_npz(RESULTS / "c10b-dgp-perturbation-draws.npz")
    scenarios = (
        "S0 baseline (fitted nu, matched)",
        "S1 lighter tails nu=8 (matched)",
        "S2 heavier tails nu=2.5 (matched)",
        "S3 near-Gaussian t-copula (nu_c=200, fitted margins)",
        "S4 MIS-SPECIFIED (simulate nu=2.5, calibrate at fitted nu)",
    )
    required = {
        "scenario_names", "scenario_is_s0_prefix", "scenario_nu_sim",
        "scenario_nu_cal", "scenario_nu_c_sim", "scenario_nu_c_cal",
        "scenario_law_sha256",
        "panel_seeds", "scenario_panel_seeds", "scenario_panel_usable_mask",
        "scenario_panel_excluded_mask", "scenario_panel_se_ok_full",
        "scenario_panel_delta_infra_full", "scenario_panel_delta_reg_full",
        "scenario_panel_se_infra_full", "scenario_panel_se_reg_full",
        "scenario_panel_dbar_full", "scenario_panel_p_iid_full",
        "scenario_panel_p_de_full", "gaussian_reference_seeds",
        "gaussian_reference_full", "gaussian_reference_usable_mask",
        "gaussian_reference_excluded_mask", "scenario_gaussian_reference_seeds",
        "scenario_gaussian_reference_full", "scenario_gaussian_reference_usable_mask",
        "scenario_gaussian_reference_excluded_mask", "t_reference_seeds",
        "scenario_t_reference_seeds", "scenario_t_reference_full",
        "scenario_t_reference_usable_mask", "scenario_t_reference_excluded_mask",
        "scenario_crit_gauss", "scenario_crit_t", "c10b_seed_schema_version",
        "fit_seed", "mc_seed", "rescue_seed", "panel_first_seed",
        "panel_last_seed", "gaussian_reference_first_seed",
        "gaussian_reference_last_seed", "t_reference_first_seed",
        "t_reference_last_seed", "N_requested", "B_ref_requested",
        "c9_seed_schema_version", "c9_exact_identity_verified",
        "c10_s0_prefix_verified", "c10_source_archive",
        "c10_source_archive_sha256", "source_c10_analysis_contract_version",
        "source_c10_analysis_sha256", "source_c10_source_sha256",
        "source_c10_parent_c9_analysis_stack_sha256",
        "source_c10_downstream_analysis_stack_sha256",
        "c10b_analysis_contract_version", "c10b_analysis_contract",
        "c10b_analysis_sha256", "c10b_scenario_laws_sha256",
        "c10b_source_sha256", "c10b_parent_c10_analysis_stack_sha256",
        "c10b_downstream_analysis_stack_sha256",
    }
    require_fields(draws, required, label)
    require_exact(draws["scenario_names"], np.asarray(scenarios),
                  "c10b scenario names/order drifted")
    require_exact(draws["scenario_is_s0_prefix"],
                  np.asarray([True, False, False, False, False]),
                  "c10b S0-prefix flags drifted")
    require(int(scalar(draws, "c10b_seed_schema_version")) == 1,
            "c10b seed schema version drifted")
    require(int(scalar(draws, "c9_seed_schema_version")) == 2 and
            bool(scalar(draws, "c9_exact_identity_verified")) and
            bool(scalar(draws, "c10_s0_prefix_verified")),
            "c10b lost a c9/c10 provenance gate")
    require(scalar_text(draws, "c10_source_archive") ==
            "results/c10-size-study-draws.npz",
            "c10b source archive is not package-relative c10")
    c10_archive_sha256 = sha256_file(RESULTS / "c10-size-study-draws.npz")
    require(scalar_text(draws, "c10_source_archive_sha256") == c10_archive_sha256,
            "c10b source c10 archive SHA-256 is stale")
    require((int(scalar(draws, "fit_seed")), int(scalar(draws, "mc_seed")),
             int(scalar(draws, "rescue_seed"))) == (FIT_SEED, MC_SEED, RESCUE_SEED),
            "c10b fit/Monte-Carlo/rescue seeds drifted")
    N = int(scalar(draws, "N_requested"))
    B_ref = int(scalar(draws, "B_ref_requested"))
    require((N, B_ref) == (300, 2000),
            "c10b final release must use N=300 and B_ref=2,000")
    expected_panel = np.arange(MC_SEED, MC_SEED + N, dtype=np.int64)
    expected_g = np.arange(MC_SEED + 700_000, MC_SEED + 700_000 + B_ref, dtype=np.int64)
    expected_t = expected_g + 500_000
    require_exact(draws["panel_seeds"], expected_panel, "c10b panel seed array drifted")
    require_exact(draws["gaussian_reference_seeds"], expected_g,
                  "c10b Gaussian reference seed array drifted")
    require_exact(draws["t_reference_seeds"], expected_t,
                  "c10b Student-t reference seed array drifted")
    require_exact(draws["scenario_panel_seeds"], np.tile(expected_panel, (5, 1)),
                  "c10b common-random-number panel seed matrix drifted")
    require_exact(draws["scenario_gaussian_reference_seeds"], np.tile(expected_g, (5, 1)),
                  "c10b Gaussian reference seed matrix drifted")
    require_exact(draws["scenario_t_reference_seeds"], np.tile(expected_t, (5, 1)),
                  "c10b Student-t reference seed matrix drifted")
    for field, expected in (
        ("panel_first_seed", expected_panel[0]), ("panel_last_seed", expected_panel[-1]),
        ("gaussian_reference_first_seed", expected_g[0]),
        ("gaussian_reference_last_seed", expected_g[-1]),
        ("t_reference_first_seed", expected_t[0]),
        ("t_reference_last_seed", expected_t[-1]),
    ):
        require(int(scalar(draws, field)) == int(expected), f"c10b {field} is inconsistent")

    panel_mask = np.asarray(draws["scenario_panel_usable_mask"], dtype=bool)
    panel_excluded = np.asarray(draws["scenario_panel_excluded_mask"], dtype=bool)
    dbar = np.asarray(draws["scenario_panel_dbar_full"], dtype=float)
    p_iid = np.asarray(draws["scenario_panel_p_iid_full"], dtype=float)
    p_de = np.asarray(draws["scenario_panel_p_de_full"], dtype=float)
    di = np.asarray(draws["scenario_panel_delta_infra_full"], dtype=float)
    dr = np.asarray(draws["scenario_panel_delta_reg_full"], dtype=float)
    se_i = np.asarray(draws["scenario_panel_se_infra_full"], dtype=float)
    se_r = np.asarray(draws["scenario_panel_se_reg_full"], dtype=float)
    se_ok = np.asarray(draws["scenario_panel_se_ok_full"], dtype=bool)
    require(panel_mask.shape == panel_excluded.shape == dbar.shape == p_iid.shape ==
            p_de.shape == se_ok.shape == (5, N), "c10b panel-level matrix shapes drifted")
    require(di.shape == dr.shape == se_i.shape == se_r.shape == (5, N, 6),
            "c10b per-asset panel tensor shapes drifted")
    require_exact(panel_excluded, ~panel_mask, "c10b panel masks are not complements")
    require_exact(panel_mask, ~np.isnan(dbar), "c10b panel masks do not identify dbar NaNs")
    require(np.all(np.isfinite(di[panel_mask])) and np.all(np.isfinite(dr[panel_mask])),
            "c10b usable panels contain non-finite coefficient values")
    require(np.all(np.isnan(di[~panel_mask])) and np.all(np.isnan(dr[~panel_mask])) and
            np.all(np.isnan(se_i[~panel_mask])) and np.all(np.isnan(se_r[~panel_mask])) and
            np.all(np.isnan(p_iid[~panel_mask])) and np.all(np.isnan(p_de[~panel_mask])),
            "c10b excluded panels do not use NaN consistently")
    expected_se_ok = panel_mask & np.all(np.isfinite(se_i) & np.isfinite(se_r), axis=2)
    require_exact(se_ok, expected_se_ok,
                  "c10b panel SE flags do not identify six finite SE pairs")
    require(not np.any(np.isinf(se_i)) and not np.any(np.isinf(se_r)) and
            not np.any(np.isinf(p_iid)) and not np.any(np.isinf(p_de)),
            "c10b SE or p-value arrays contain infinities")
    for i, scenario in enumerate(scenarios):
        require(np.allclose(dbar[i, panel_mask[i]],
                            np.mean(di[i, panel_mask[i]] - dr[i, panel_mask[i]], axis=1),
                            rtol=0.0, atol=1e-12),
                f"c10b {scenario} dbar/per-asset differences disagree")

    g_full = np.asarray(draws["gaussian_reference_full"], dtype=float)
    g_mask = np.asarray(draws["gaussian_reference_usable_mask"], dtype=bool)
    g_excluded = np.asarray(draws["gaussian_reference_excluded_mask"], dtype=bool)
    sg_full = np.asarray(draws["scenario_gaussian_reference_full"], dtype=float)
    sg_mask = np.asarray(draws["scenario_gaussian_reference_usable_mask"], dtype=bool)
    sg_excluded = np.asarray(draws["scenario_gaussian_reference_excluded_mask"], dtype=bool)
    require(g_full.shape == g_mask.shape == g_excluded.shape == (B_ref,),
            "c10b Gaussian reference shapes drifted")
    require(not np.any(np.isinf(g_full)),
            "c10b Gaussian reference vector contains infinities")
    require_exact(g_mask, ~np.isnan(g_full), "c10b Gaussian usable mask disagrees")
    require_exact(g_excluded, ~g_mask, "c10b Gaussian masks are not complements")
    require_exact(sg_full, np.tile(g_full, (5, 1)),
                  "c10b scenario Gaussian references are not one cached law")
    require_exact(sg_mask, np.tile(g_mask, (5, 1)),
                  "c10b scenario Gaussian masks are not identical")
    require_exact(sg_excluded, ~sg_mask, "c10b scenario Gaussian masks are not complements")

    t_full = np.asarray(draws["scenario_t_reference_full"], dtype=float)
    t_mask = np.asarray(draws["scenario_t_reference_usable_mask"], dtype=bool)
    t_excluded = np.asarray(draws["scenario_t_reference_excluded_mask"], dtype=bool)
    require(t_full.shape == t_mask.shape == t_excluded.shape == (5, B_ref),
            "c10b Student-t reference matrix shapes drifted")
    require(not np.any(np.isinf(t_full)),
            "c10b Student-t reference matrix contains infinities")
    require_exact(t_mask, ~np.isnan(t_full), "c10b Student-t usable masks disagree")
    require_exact(t_excluded, ~t_mask, "c10b Student-t masks are not complements")
    require_exact(t_full[0], t_full[4],
                  "c10b S0 and S4 do not reuse the fitted-nu calibration law")
    require_exact(t_mask[0], t_mask[4],
                  "c10b S0 and S4 fitted-nu calibration masks differ")
    crit_g = np.asarray(draws["scenario_crit_gauss"], dtype=float)
    crit_t = np.asarray(draws["scenario_crit_t"], dtype=float)
    require(crit_g.shape == crit_t.shape == (5, 2),
            "c10b scenario critical-value matrices have wrong shape")
    for i, scenario in enumerate(scenarios):
        require(np.allclose(crit_g[i], np.percentile(g_full[g_mask], [95, 90]),
                            rtol=0.0, atol=1e-12),
                f"c10b {scenario} Gaussian critical values are inconsistent")
        require(np.allclose(crit_t[i], np.percentile(t_full[i, t_mask[i]], [95, 90]),
                            rtol=0.0, atol=1e-12),
                f"c10b {scenario} Student-t critical values are inconsistent")

    nu_sim = np.asarray(draws["scenario_nu_sim"], dtype=float)
    nu_cal = np.asarray(draws["scenario_nu_cal"], dtype=float)
    nu_c_sim = np.asarray(draws["scenario_nu_c_sim"], dtype=float)
    nu_c_cal = np.asarray(draws["scenario_nu_c_cal"], dtype=float)
    fitted_nu = np.asarray(draws["nu_null_by_asset"], dtype=float)
    fitted_nu_c = float(scalar(draws, "nu_c_null"))
    expected_nu_sim = np.vstack([fitted_nu, np.full(6, 8.0), np.full(6, 2.5),
                                 fitted_nu, np.full(6, 2.5)])
    expected_nu_cal = np.vstack([fitted_nu, np.full(6, 8.0), np.full(6, 2.5),
                                 fitted_nu, fitted_nu])
    require_exact(nu_sim, expected_nu_sim, "c10b simulation-margin scenario definitions drifted")
    require_exact(nu_cal, expected_nu_cal, "c10b calibration-margin scenario definitions drifted")
    require_exact(nu_c_sim, np.asarray([fitted_nu_c, 8.0, 2.5, 200.0, 2.5]),
                  "c10b simulation copula-df scenario definitions drifted")
    require_exact(nu_c_cal, np.asarray([fitted_nu_c, 8.0, 2.5, 200.0, fitted_nu_c]),
                  "c10b calibration copula-df scenario definitions drifted")

    c10_parent_provenance = {
        key: scalar(c10_draws, key)
        for key in (
            "c10_analysis_contract_version", "c10_analysis_sha256",
            "c10_source_sha256", "c10_parent_c9_analysis_stack_sha256",
            "c10_downstream_analysis_stack_sha256",
        )
    }
    for local_key, parent_key in (
        ("source_c10_analysis_contract_version", "c10_analysis_contract_version"),
        ("source_c10_analysis_sha256", "c10_analysis_sha256"),
        ("source_c10_source_sha256", "c10_source_sha256"),
        ("source_c10_parent_c9_analysis_stack_sha256",
         "c10_parent_c9_analysis_stack_sha256"),
        ("source_c10_downstream_analysis_stack_sha256",
         "c10_downstream_analysis_stack_sha256"),
    ):
        require(str(scalar(draws, local_key)) == str(c10_parent_provenance[parent_key]),
                f"c10b {local_key} differs from its source c10 archive")
    c10b_provenance = c10b.current_c10b_analysis_provenance(
        c10b._scenario_definitions(fitted_nu),
        c10_parent_provenance["c10_downstream_analysis_stack_sha256"],
    )
    row_law_hashes = np.asarray(
        c10b_provenance["c10b_scenario_law_sha256_by_row"]
    )
    require_exact(draws["scenario_law_sha256"], row_law_hashes,
                  "c10b per-row scenario-law hashes differ from current definitions")
    for i, digest in enumerate(row_law_hashes):
        require_hash(str(digest), f"c10b scenario {i} law hash")
    c10b_scalar_provenance = {
        key: value for key, value in c10b_provenance.items()
        if key != "c10b_scenario_law_sha256_by_row"
    }
    require_current_provenance(draws, c10b_scalar_provenance,
                               "c10b downstream provenance")

    # S0 is not an independent rerun: it must be the exact c10 prefix.
    c10_panel_mask = np.asarray(c10_draws["panel_usable_mask"], dtype=bool)[:N]
    prefix_pairs = (
        (panel_mask[0], c10_panel_mask, "panel usable mask"),
        (se_ok[0], np.asarray(c10_draws["panel_se_ok_full"])[:N], "panel SE flag"),
        (di[0], np.asarray(c10_draws["panel_delta_infra_full"])[:N], "delta_infra"),
        (dr[0], np.asarray(c10_draws["panel_delta_reg_full"])[:N], "delta_reg"),
        (se_i[0], np.asarray(c10_draws["panel_se_infra_full"])[:N], "se_infra"),
        (se_r[0], np.asarray(c10_draws["panel_se_reg_full"])[:N], "se_reg"),
        (dbar[0], np.asarray(c10_draws["panel_dbar_full"])[:N], "dbar"),
        (p_iid[0], np.asarray(c10_draws["panel_p_iid_full"])[:N], "iid p"),
        (p_de[0], np.asarray(c10_draws["panel_p_de_full"])[:N], "design-effect p"),
        (g_full, np.asarray(c10_draws["gaussian_reference_full"])[:B_ref],
         "Gaussian reference"),
        (g_mask, np.asarray(c10_draws["gaussian_reference_usable_mask"])[:B_ref],
         "Gaussian reference usable mask"),
        (t_full[0], np.asarray(c10_draws["t_reference_full"])[:B_ref],
         "Student-t reference"),
        (t_mask[0], np.asarray(c10_draws["t_reference_usable_mask"])[:B_ref],
         "Student-t reference usable mask"),
    )
    for actual, expected, name in prefix_pairs:
        require_exact(actual, expected, f"c10b S0 {name} is not the exact c10 prefix")

    _validate_union_calendar(draws, "", label)
    _require_baseline_dgp_identity(draws, c9_baseline, label)
    csv = pd.read_csv(csv_path)
    required_csv = {
        "scenario", "N_used", "n_calendar", "availability_pattern_codes",
        "availability_pattern_counts", "fit_seed", "mc_seed", "rescue_seed",
        "crn_policy", "design_sha256", "fixed_null_dgp_sha256",
        "refitter_sha256", "simulator_sha256", "analysis_stack_sha256",
        "source_c10_archive_sha256", "source_c10_analysis_sha256",
        "source_c10_source_sha256", "source_c10_downstream_analysis_stack_sha256",
        "c10b_analysis_contract_version", "c10b_analysis_sha256",
        "c10b_scenario_laws_sha256", "scenario_law_sha256",
        "c10b_source_sha256",
        "c10b_parent_c10_analysis_stack_sha256",
        "c10b_downstream_analysis_stack_sha256",
    }
    for method in ("naive_iid", "design_effect", "gaussian_copula_boot", "tcopula_boot"):
        required_csv.update({f"{method}_size05", f"{method}_se05", f"{method}_size10"})
    require_fields(csv.columns, required_csv, "c10b results CSV")
    rows = indexed_rows(csv, "scenario", scenarios, "c10b results CSV")
    for i, scenario in enumerate(scenarios):
        row = rows.loc[scenario]
        used = int(panel_mask[i].sum())
        _csv_int(row, "N_used", used, f"c10b {scenario}")
        for field, expected in (("n_calendar", 2434), ("fit_seed", FIT_SEED),
                                ("mc_seed", MC_SEED), ("rescue_seed", RESCUE_SEED)):
            _csv_int(row, field, expected, f"c10b {scenario}")
        require(str(row["availability_pattern_codes"]) == "55;63" and
                str(row["availability_pattern_counts"]) == "243;2191" and
                str(row["crn_policy"]) == "same seed positions across all scenarios",
                f"c10b {scenario} calendar/CRN disclosure drifted")
        decisions = {
            "naive_iid": (p_iid[i, panel_mask[i]] < 0.05,
                          p_iid[i, panel_mask[i]] < 0.10),
            "design_effect": (p_de[i, panel_mask[i]] < 0.05,
                              p_de[i, panel_mask[i]] < 0.10),
            "gaussian_copula_boot": (dbar[i, panel_mask[i]] > crit_g[i, 0],
                                      dbar[i, panel_mask[i]] > crit_g[i, 1]),
            "tcopula_boot": (dbar[i, panel_mask[i]] > crit_t[i, 0],
                              dbar[i, panel_mask[i]] > crit_t[i, 1]),
        }
        for method, (reject5, reject10) in decisions.items():
            size5 = float(np.mean(reject5))
            size10 = float(np.mean(reject10))
            _csv_float(row, f"{method}_size05", size5, f"c10b {scenario}")
            _csv_float(row, f"{method}_size10", size10, f"c10b {scenario}")
            _csv_float(row, f"{method}_se05", np.sqrt(size5 * (1 - size5) / used),
                       f"c10b {scenario}")
        for csv_key, archive_key in (
            ("design_sha256", "audit_design_sha256"),
            ("fixed_null_dgp_sha256", "audit_fixed_null_dgp_sha256"),
            ("refitter_sha256", "audit_refitter_sha256"),
            ("simulator_sha256", "audit_simulator_sha256"),
            ("analysis_stack_sha256", "audit_analysis_stack_sha256"),
        ):
            require(str(row[csv_key]) == scalar_text(draws, archive_key),
                    f"c10b {scenario} CSV {csv_key} differs from archive")
        for csv_key, expected in (
            ("source_c10_archive_sha256", c10_archive_sha256),
            ("source_c10_analysis_sha256",
             c10_parent_provenance["c10_analysis_sha256"]),
            ("source_c10_source_sha256", c10_parent_provenance["c10_source_sha256"]),
            ("source_c10_downstream_analysis_stack_sha256",
             c10_parent_provenance["c10_downstream_analysis_stack_sha256"]),
            *(
                (key, value)
                for key, value in c10b_scalar_provenance.items()
                if key != "c10b_analysis_contract"
            ),
            ("scenario_law_sha256", row_law_hashes[i]),
        ):
            require(str(row[csv_key]) == str(expected),
                    f"c10b {scenario} CSV {csv_key} differs from current provenance")
    finding_snippets = []
    for scenario in scenarios:
        row = rows.loc[scenario]
        finding_snippets.append((
            f"{scenario} result row",
            f"| {scenario} | {int(row['N_used'])} | "
            f"{float(row['naive_iid_size05']):.3f} | "
            f"{float(row['design_effect_size05']):.3f} | "
            f"{float(row['gaussian_copula_boot_size05']):.3f} | "
            f"{float(row['tcopula_boot_size05']):.3f} |",
        ))
    finding_snippets.append((
        "design-effect column label",
        "| scenario | usable | naive iid | design-effect t(5) | "
        "Gaussian comparator | Student-t copula |",
    ))
    finding_path = RESULTS / "c10b-dgp-perturbation-FINDING.md"
    require_finding(
        finding_path,
        "# C10b -- DGP-perturbation sensitivity",
        finding_snippets,
        "c10b",
    )
    require("effective-df sensitivity" not in finding_path.read_text(),
            "c10b finding mislabels its design-effect t(5) column")
    return rows, draws


def validate_c13(c9_baseline, c10_draws):
    label = "c13"
    draws = load_npz(RESULTS / "c13-rung4-perasset-draws.npz")
    required = {
        "D", "D_full", "dbar_full", "usable_mask", "excluded_mask",
        "draw_seeds", "rescue_attempts_by_draw", "rescue_successes_by_draw",
        "R_d", "Cov_d", "d_obs", "rho_return", "rho_resid", "rho_d_bar",
        "base_seed", "first_draw_seed", "last_draw_seed", "seed_schema_version",
        "exact_c9_baseline_alignment", "n_draws_with_six_start_rescue",
        "n_asset_rescue_attempts", "n_asset_rescue_successes",
        "n_post_rescue_exclusions", *HASH_KEYS,
        "audit_simulation_calendar_ns", "audit_calendar_pos_concat",
        "audit_calendar_pos_offsets", "audit_availability_pattern_codes",
        "audit_availability_pattern_counts",
        "source_c9_baseline_archive", "source_c9_baseline_archive_sha256",
        "source_c10_archive", "source_c10_archive_sha256",
        "source_c10_analysis_contract_version", "source_c10_analysis_sha256",
        "source_c10_source_sha256",
        "source_c10_downstream_analysis_stack_sha256",
        "c13_analysis_contract_version", "c13_analysis_contract",
        "c13_analysis_sha256", "c13_source_sha256",
        "c13_parent_c9_analysis_stack_sha256",
        "c13_parent_c10_analysis_stack_sha256",
        "c13_downstream_analysis_stack_sha256",
    }
    require_fields(draws, required, label)
    require(int(scalar(draws, "seed_schema_version")) == 2 and
            bool(scalar(draws, "exact_c9_baseline_alignment")),
            "c13 lost its exact c9 schema-v2 alignment gate")
    require((int(scalar(draws, "base_seed")), int(scalar(draws, "first_draw_seed")),
             int(scalar(draws, "last_draw_seed"))) ==
            (FIT_SEED, FIT_SEED + DRAW_SEED_OFFSET,
             FIT_SEED + DRAW_SEED_OFFSET + 1999),
            "c13 seed summary drifted")
    expected_seeds = np.arange(FIT_SEED + DRAW_SEED_OFFSET,
                               FIT_SEED + DRAW_SEED_OFFSET + 2000,
                               dtype=np.int64)
    require_exact(draws["draw_seeds"], expected_seeds, "c13 draw-seed array drifted")
    require_exact(draws["draw_seeds"], c9_baseline["draw_seeds"],
                  "c13/c9 baseline draw-seed arrays differ")
    D_full = np.asarray(draws["D_full"], dtype=float)
    usable = np.asarray(draws["usable_mask"], dtype=bool)
    excluded = np.asarray(draws["excluded_mask"], dtype=bool)
    dbar = np.asarray(draws["dbar_full"], dtype=float)
    require(D_full.shape == (2000, 6) and usable.shape == excluded.shape ==
            dbar.shape == (2000,), "c13 full draw/mask shapes drifted")
    require_exact(usable, ~np.isnan(D_full).any(axis=1),
                  "c13 usable mask does not identify complete per-asset rows")
    require_exact(excluded, ~usable, "c13 masks are not complements")
    require_exact(draws["D"], D_full[usable], "c13 compressed/full per-asset draws disagree")
    require_exact(dbar, np.mean(D_full, axis=1), "c13 dbar is not the mean of D_full")
    require_exact(usable, c9_baseline["null_t_usable_mask"],
                  "c13/c9 baseline usable masks differ")
    require_exact(dbar, c9_baseline["null_t_full"],
                  "c13/c9 baseline statistics differ seed-for-seed")
    require_exact(dbar[usable], c9_baseline["null_t"],
                  "c13/c9 compressed baseline statistics differ")

    attempts = np.asarray(draws["rescue_attempts_by_draw"], dtype=np.int64)
    successes = np.asarray(draws["rescue_successes_by_draw"], dtype=np.int64)
    require(attempts.shape == successes.shape == (2000,),
            "c13 rescue telemetry shapes drifted")
    require(np.all(attempts >= 0) and np.all(successes >= 0) and
            np.all(successes <= attempts), "c13 rescue telemetry is invalid")
    require_exact(attempts, c9_baseline["n_asset_refits_with_six_start_rescue_attempt_t"],
                  "c13/c9 asset rescue-attempt telemetry differs")
    require_exact(successes, c9_baseline["n_asset_refits_rescued_by_six_start_t"],
                  "c13/c9 asset rescue-success telemetry differs")
    for field, expected in (
        ("n_draws_with_six_start_rescue", np.sum(attempts > 0)),
        ("n_asset_rescue_attempts", attempts.sum()),
        ("n_asset_rescue_successes", successes.sum()),
        ("n_post_rescue_exclusions", excluded.sum()),
    ):
        require(int(scalar(draws, field)) == int(expected), f"c13 {field} is inconsistent")

    D = D_full[usable]
    expected_R = np.corrcoef(D, rowvar=False)
    expected_cov = np.cov(D, rowvar=False)
    require(np.allclose(draws["R_d"], expected_R, rtol=0.0, atol=1e-12),
            "c13 R_d does not equal corr(D)")
    require(np.allclose(draws["Cov_d"], expected_cov, rtol=0.0, atol=1e-12),
            "c13 Cov_d does not equal cov(D)")
    expected_rho = (expected_R.sum() - np.trace(expected_R)) / 30.0
    require_near(scalar(draws, "rho_d_bar"), expected_rho,
                 "c13 rho_d_bar does not equal the mean off-diagonal of R_d")
    c9_R_z = np.asarray(c9_baseline["audit_R_z"], dtype=float)
    c9_rho_resid = (c9_R_z.sum() - np.trace(c9_R_z)) / 30.0
    require_near(scalar(draws, "rho_resid"), c9_rho_resid,
                 "c13 rho_resid does not equal c9's mean off-diagonal R_z")
    d_obs = np.asarray(draws["d_obs"], dtype=float)
    require(d_obs.shape == (6,), "c13 observed per-asset contrast has wrong shape")
    require_near(d_obs.mean(), scalar(c9_baseline, "d_bar_obs"),
                 "c13/c9 observed dbar differs", tol=0.0)
    for key in HASH_KEYS:
        require(scalar_text(draws, key) == scalar_text(c9_baseline, key),
                f"c13/c9 {key} differs")
    for local_key, c9_key in (
        ("audit_simulation_calendar_ns", "audit_simulation_calendar_ns"),
        ("audit_calendar_pos_concat", "audit_calendar_pos_concat"),
        ("audit_calendar_pos_offsets", "audit_calendar_pos_offsets"),
        ("audit_availability_pattern_codes", "audit_availability_pattern_codes"),
        ("audit_availability_pattern_counts", "audit_availability_pattern_counts"),
    ):
        require_exact(draws[local_key], c9_baseline[c9_key],
                      f"c13/c9 {local_key} differs")
    require(scalar_text(draws, "source_c9_baseline_archive") ==
            "results/c9-tcopula-draws-baseline.npz",
            "c13 source c9 archive label drifted")
    require(scalar_text(draws, "source_c9_baseline_archive_sha256") ==
            sha256_file(RESULTS / "c9-tcopula-draws-baseline.npz"),
            "c13 source c9 archive SHA-256 is stale")
    require(scalar_text(draws, "source_c10_archive") ==
            "results/c10-size-study-draws.npz",
            "c13 source c10 archive label drifted")
    c10_archive_sha256 = sha256_file(RESULTS / "c10-size-study-draws.npz")
    require(scalar_text(draws, "source_c10_archive_sha256") == c10_archive_sha256,
            "c13 source c10 archive SHA-256 is stale")
    c10_parent_provenance = {
        key: scalar(c10_draws, key)
        for key in (
            "c10_analysis_contract_version", "c10_analysis_sha256",
            "c10_source_sha256", "c10_downstream_analysis_stack_sha256",
        )
    }
    for local_key, parent_key in (
        ("source_c10_analysis_contract_version", "c10_analysis_contract_version"),
        ("source_c10_analysis_sha256", "c10_analysis_sha256"),
        ("source_c10_source_sha256", "c10_source_sha256"),
        ("source_c10_downstream_analysis_stack_sha256",
         "c10_downstream_analysis_stack_sha256"),
    ):
        require(str(scalar(draws, local_key)) == str(c10_parent_provenance[parent_key]),
                f"c13 {local_key} differs from its source c10 archive")
    c13_provenance = c13.current_c13_analysis_provenance(
        scalar_text(c9_baseline, "audit_analysis_stack_sha256"),
        c10_parent_provenance["c10_downstream_analysis_stack_sha256"],
    )
    require_current_provenance(draws, c13_provenance, "c13 downstream provenance")
    result_path = RESULTS / "c13-rung4-recompute-results.csv"
    require(result_path.is_file(), "c13 results CSV is missing")
    text = result_path.read_text()

    def section(after, before=None):
        require(after in text, f"c13 results file lacks section marker: {after}")
        body = text.split(after, 1)[1]
        if before is not None:
            require(before in body, f"c13 results file lacks section marker: {before}")
            body = body.split(before, 1)[0]
        return pd.read_csv(io.StringIO(body.strip()))

    marker_rung = "# rung-4 dispersion statistic under alternative rho inputs and references\n"
    marker_mc = "# #MC: fitted-DGP rejection rates on c10 draws under each reference\n"
    comment_metadata = dict(re.findall(r"(?m)^# ([a-z0-9_]+)=([^\n]+)$", text))
    for key, expected in (
        ("c13_analysis_sha256", c13_provenance["c13_analysis_sha256"]),
        ("c13_source_sha256", c13_provenance["c13_source_sha256"]),
        ("c13_downstream_analysis_stack_sha256",
         c13_provenance["c13_downstream_analysis_stack_sha256"]),
        ("source_c10_archive_sha256", c10_archive_sha256),
    ):
        require(comment_metadata.get(key) == str(expected),
                f"c13 results CSV comment {key} differs from current provenance")
    rung = section(marker_rung, marker_mc)
    mc = section(marker_mc)
    rho_labels = (
        "rho_return (c6 legacy input)",
        "rho_resid (standardized-residual sensitivity)",
        "rho_d_bar (fitted-null estimator-difference sensitivity)",
    )
    rung = indexed_rows(rung, "rho_label", rho_labels, "c13 rung-4 table")
    mean_d = float(d_obs.mean())
    N_assets = len(d_obs)
    se_naive = float(d_obs.std(ddof=1) / np.sqrt(N_assets))
    rho_values = (
        float(scalar(draws, "rho_return")),
        float(scalar(draws, "rho_resid")),
        float(scalar(draws, "rho_d_bar")),
    )
    for label_name, rho in zip(rho_labels, rho_values):
        row = rung.loc[label_name]
        deff = 1 + (N_assets - 1) * rho
        n_eff = N_assets / deff
        df_eff = (N_assets - 1) / deff
        se_de = se_naive * np.sqrt(deff)
        t_value = mean_d / se_de
        expected = {
            "rho": rho, "DEFF": deff, "N_eff": n_eff, "df_eff": df_eff,
            "se_de": se_de, "t_dispersion": t_value,
            "p_t_dfeff_dispersion": stats.t.sf(t_value, df=df_eff),
            "p_t_N1_dispersion": stats.t.sf(t_value, df=N_assets - 1),
            "p_norm_dispersion": stats.norm.sf(t_value),
        }
        for field, value in expected.items():
            _csv_float(row, field, value, f"c13 {label_name}")

    p_de = np.asarray(c10_draws["p_de"], dtype=float)
    p_de = p_de[np.isfinite(p_de)]
    t_panel = stats.t.isf(p_de, df=N_assets - 1)
    df_eff_return = (N_assets - 1) / (1 + (N_assets - 1) * rho_values[0])
    df_eff_d = (N_assets - 1) / (1 + (N_assets - 1) * rho_values[2])
    expected_crit = (
        (stats.norm.isf(0.05), stats.norm.isf(0.10)),
        (stats.t.isf(0.05, N_assets - 1), stats.t.isf(0.10, N_assets - 1)),
        (stats.t.isf(0.05, df_eff_return), stats.t.isf(0.10, df_eff_return)),
        (stats.t.isf(0.05, df_eff_d), stats.t.isf(0.10, df_eff_d)),
    )
    expected_mc_labels = [
        "normal z (size study's z=1.645)",
        "t(N-1=5) [c10 saved rule]",
        f"t(df_eff={df_eff_return:.2f}) [sensitivity, rho_return]",
        f"t(df_eff={df_eff_d:.2f}) [sensitivity, rho_d_bar]",
    ]
    require(len(mc) == 4 and mc["crit_label"].tolist() == expected_mc_labels,
            "c13 conditional-size table row names/order drifted")
    for i, row in mc.iterrows():
        crit5, crit10 = expected_crit[i]
        for field, value in (
            ("crit_005", crit5), ("crit_010", crit10),
            ("size_005", np.mean(t_panel > crit5)),
            ("size_010", np.mean(t_panel > crit10)),
        ):
            _csv_float(row, field, value, f"c13 conditional-size row {i}")
        _csv_int(row, "n_panels", len(t_panel), f"c13 conditional-size row {i}")
    finding_snippets = []
    for label_name in rho_labels:
        row = rung.loc[label_name]
        finding_snippets.append((
            f"{label_name} analytical row",
            f"| {label_name} | {float(row['rho']):.3f} | "
            f"{float(row['DEFF']):.2f} | {float(row['N_eff']):.2f} | "
            f"{float(row['df_eff']):.2f} | {float(row['t_dispersion']):.2f} | "
            f"{float(row['p_t_dfeff_dispersion']):.4f} | "
            f"{float(row['p_t_N1_dispersion']):.4f} | "
            f"{float(row['p_norm_dispersion']):.4f} |",
        ))
    for i, row in mc.iterrows():
        display_label = str(row["crit_label"]).replace("[corrected,", "[sensitivity,")
        finding_snippets.append((
            f"conditional-size row {i}",
            f"| {display_label} | {float(row['crit_005']):.3f} | "
            f"{float(row['crit_010']):.3f} | {float(row['size_005']):.3f} | "
            f"{float(row['size_010']):.3f} | {int(row['n_panels'])} |",
        ))
    effective_df_p = rung["p_t_dfeff_dispersion"].to_numpy(dtype=float)
    finding_snippets.append((
        "effective-df dispersion span",
        "Across the three dependence inputs for the dispersion statistic under "
        "the t(df_eff) convention, the descriptive p-value span is "
        f"{effective_df_p.min():.3f}-{effective_df_p.max():.3f}.",
    ))
    finding_path = RESULTS / "c13-rung4-recompute-FINDING.md"
    require_finding(
        finding_path,
        "# C13 -- Rung-4 design-effect sensitivity diagnostic",
        finding_snippets,
        "c13",
    )
    finding_text = finding_path.read_text()
    for stale_token in ("supplied statistic", "2.328", "1.763"):
        require(stale_token not in finding_text,
                f"c13 finding retains deleted stale token: {stale_token}")
    return draws


def validate_c19(c9_payloads):
    outputs = {}
    current_recursive_contract = _current_c19_contract()
    current_recursive_hash = hashlib.sha256(
        current_recursive_contract.encode("utf-8")
    ).hexdigest()
    for spec in ("baseline", "crisis"):
        label = f"c19 {spec}"
        csv_path = RESULTS / f"c19-recursive-sensitivity-{spec}.csv"
        require(csv_path.is_file(), f"required {label} CSV is missing")
        frame = pd.read_csv(csv_path)
        require(len(frame) == 1, f"{label} CSV must contain exactly one row")
        row = frame.iloc[0]
        required_csv = {
            "spec", "base_seed", "recursive_seed_schema_version",
            "recursive_draw_seed_offset", "first_draw_seed", "last_draw_seed",
            "design_sha256", "fixed_null_dgp_sha256", "refitter_sha256",
            "simulator_sha256", "analysis_stack_sha256",
            "recursive_simulator_sha256", "recursive_analysis_stack_sha256",
            "n_calendar", "availability_pattern_codes", "availability_pattern_counts",
            "B", "B_requested", "B_used", "n_total_exclusions",
            "n_recursion_explosion_exclusions",
            "n_refit_exclusions_after_six_start_rescue",
            "n_draws_with_six_start_rescue_attempt",
            "n_draws_retained_after_six_start_rescue",
            "n_draws_failed_after_six_start_rescue",
            "n_asset_refits_with_six_start_rescue_attempt",
            "n_asset_refits_rescued_by_six_start", "fraction_total_excluded",
            "fraction_recursion_explosion_excluded",
            "fraction_refit_excluded_after_six_start_rescue", "d_bar_obs",
            "n_upper_tail_hits_recursive", "n_abs_stat_hits_recursive",
            "n_lower_tail_hits_recursive", "p_one_recursive", "p_abs_recursive",
            "p_equal_tail_recursive", "p_worst_low", "p_worst_high",
            "null_mean_recursive", "null_sd_recursive", "B_used_fixed",
            "B_requested_fixed", "n_total_exclusions_fixed",
            "fraction_total_excluded_fixed",
            "n_draws_with_six_start_rescue_attempt_fixed",
            "n_draws_retained_after_six_start_rescue_fixed",
            "n_draws_failed_after_six_start_rescue_fixed",
            "n_asset_refits_with_six_start_rescue_attempt_fixed",
            "n_asset_refits_rescued_by_six_start_fixed", "design_sha256_fixed",
            "fixed_null_dgp_sha256_fixed", "refitter_sha256_fixed",
            "simulator_sha256_fixed", "analysis_stack_sha256_fixed",
            "fixed_audit_locally_reconstructed_equal", "n_upper_tail_hits_fixed",
            "n_abs_stat_hits_fixed", "n_lower_tail_hits_fixed", "p_one_fixed",
            "p_abs_fixed", "p_equal_tail_fixed", "null_mean_fixed", "null_sd_fixed",
            "realized_var_median_recursive", "sample_var_mean_actual", "runtime_s",
            "n_draws_with_variance_floor_hit", "fraction_draws_with_variance_floor_hit",
            "n_variance_floor_hits", "n_asset_paths_with_variance_floor_hit",
            "min_raw_variance_before_floor", "variance_floor_hits_by_asset",
            "draws_with_variance_floor_hit_by_asset", "min_raw_variance_by_asset",
            "n_usable_draws_with_variance_floor_hit",
            "n_upper_tail_hits_recursive_with_variance_floor_hit",
            "n_abs_stat_hits_recursive_with_variance_floor_hit",
            "null_mean_recursive_with_variance_floor_hit",
            "null_sd_recursive_with_variance_floor_hit",
            "n_usable_draws_without_variance_floor_hit",
            "n_upper_tail_hits_recursive_without_variance_floor_hit",
            "n_abs_stat_hits_recursive_without_variance_floor_hit",
            "null_mean_recursive_without_variance_floor_hit",
            "null_sd_recursive_without_variance_floor_hit",
        }
        require_fields(frame.columns, required_csv, f"{label} CSV")
        require(str(row["spec"]) == spec, f"{label} CSV specification label drifted")

        draws = load_npz(RESULTS / f"c19-recursive-draws-{spec}.npz")
        required = {
            "dbar", "draw_seeds", "recursive_usable_mask", "recursive_excluded_mask",
            "B_requested", "exploded", "recursion_explosion_excluded",
            "draw_had_six_start_rescue_attempt", "draw_retained_after_six_start_rescue",
            "n_asset_refits_with_six_start_rescue_attempt",
            "n_asset_refits_rescued_by_six_start", "max_sig2", "mean_rv",
            "sample_vars_actual", "sample_var_mean_actual",
            "draw_had_variance_floor_hit", "n_variance_floor_hits",
            "n_asset_paths_with_variance_floor_hit", "min_raw_variance_before_floor",
            "variance_floor_hits_by_asset", "min_raw_variance_by_asset",
            "variance_floor_asset_names", "d_bar_obs", "null_fixed",
            "null_fixed_full", "null_fixed_usable_mask", "null_fixed_excluded_mask",
            "fixed_draw_seeds", "fixed_seed_schema_version", "fixed_base_seed",
            "fixed_draw_seed_offset", "fixed_first_draw_seed", "fixed_last_draw_seed",
            "fixed_draw_had_six_start_rescue_attempt",
            "fixed_draw_retained_after_six_start_rescue",
            "fixed_n_asset_refits_with_six_start_rescue_attempt",
            "fixed_n_asset_refits_rescued_by_six_start",
            "fixed_audit_design_sha256", "fixed_audit_fixed_null_dgp_sha256",
            "fixed_audit_refitter_sha256", "fixed_audit_simulator_sha256",
            "fixed_audit_analysis_stack_sha256",
            "fixed_audit_locally_reconstructed_equal", "base_seed",
            "first_draw_seed", "last_draw_seed", "recursive_seed_schema_version",
            "recursive_seed_schema", "recursive_draw_seed_offset",
            "recursive_simulator_sha256", "recursive_analysis_stack_sha256",
            "recursive_simulator_contract", *HASH_KEYS,
        }
        require_fields(draws, required, label)
        c9 = c9_payloads[spec]
        expected_seeds = np.arange(FIT_SEED + DRAW_SEED_OFFSET,
                                   FIT_SEED + DRAW_SEED_OFFSET + 2000,
                                   dtype=np.int64)
        require((int(scalar(draws, "base_seed")),
                 int(scalar(draws, "recursive_draw_seed_offset")),
                 int(scalar(draws, "B_requested"))) ==
                (FIT_SEED, DRAW_SEED_OFFSET, 2000),
                f"{label} recursive seed/request contract drifted")
        require(int(scalar(draws, "recursive_seed_schema_version")) == 1,
                f"{label} recursive seed schema version drifted")
        require_exact(draws["draw_seeds"], expected_seeds,
                      f"{label} recursive draw seeds drifted")
        require((int(scalar(draws, "first_draw_seed")),
                 int(scalar(draws, "last_draw_seed"))) ==
                (int(expected_seeds[0]), int(expected_seeds[-1])),
                f"{label} first/last recursive seed fields disagree")

        # The complete fixed reference embedded in c19 must be c9, not merely similar.
        fixed_pairs = (
            ("null_fixed", "null_t"), ("null_fixed_full", "null_t_full"),
            ("null_fixed_usable_mask", "null_t_usable_mask"),
            ("null_fixed_excluded_mask", "null_t_excluded_mask"),
            ("fixed_draw_seeds", "draw_seeds"),
            ("fixed_draw_had_six_start_rescue_attempt",
             "draw_had_six_start_rescue_attempt_t"),
            ("fixed_draw_retained_after_six_start_rescue",
             "draw_retained_after_six_start_rescue_t"),
            ("fixed_n_asset_refits_with_six_start_rescue_attempt",
             "n_asset_refits_with_six_start_rescue_attempt_t"),
            ("fixed_n_asset_refits_rescued_by_six_start",
             "n_asset_refits_rescued_by_six_start_t"),
        )
        for local_key, c9_key in fixed_pairs:
            require_exact(draws[local_key], c9[c9_key],
                          f"{label} embedded fixed {local_key} differs from c9")
        require((int(scalar(draws, "fixed_seed_schema_version")),
                 int(scalar(draws, "fixed_base_seed")),
                 int(scalar(draws, "fixed_draw_seed_offset")),
                 int(scalar(draws, "fixed_first_draw_seed")),
                 int(scalar(draws, "fixed_last_draw_seed"))) ==
                (2, FIT_SEED, DRAW_SEED_OFFSET,
                 int(expected_seeds[0]), int(expected_seeds[-1])),
                f"{label} embedded fixed seed summary drifted")
        require(bool(scalar(draws, "fixed_audit_locally_reconstructed_equal")),
                f"{label} lacks local fixed-audit equality certification")
        for key in HASH_KEYS:
            require(scalar_text(draws, key) == scalar_text(c9, key),
                    f"{label}/c9 {key} differs")
        for fixed_key, c9_key in (
            ("fixed_audit_design_sha256", "audit_design_sha256"),
            ("fixed_audit_fixed_null_dgp_sha256", "audit_fixed_null_dgp_sha256"),
            ("fixed_audit_refitter_sha256", "audit_refitter_sha256"),
            ("fixed_audit_simulator_sha256", "audit_simulator_sha256"),
            ("fixed_audit_analysis_stack_sha256", "audit_analysis_stack_sha256"),
        ):
            require(scalar_text(draws, fixed_key) == scalar_text(c9, c9_key),
                    f"{label} {fixed_key} differs from c9")
        audit_keys = {key for key in c9 if key.startswith("audit_")}
        require_fields(draws, audit_keys, f"{label} embedded c9 audit")
        for key in sorted(audit_keys):
            require_exact(draws[key], c9[key], f"{label}/c9 archived audit field {key} differs")

        recursive_contract = scalar_text(draws, "recursive_simulator_contract")
        require(recursive_contract == current_recursive_contract,
                f"{label} recursive contract does not match the current packaged source")
        recursive_sim_hash = hashlib.sha256(recursive_contract.encode()).hexdigest()
        require(recursive_sim_hash == current_recursive_hash,
                f"{label} recursive simulator digest differs from current source")
        require(scalar_text(draws, "recursive_simulator_sha256") == recursive_sim_hash,
                f"{label} recursive simulator hash does not fingerprint its contract")
        recursive_analysis_hash = hashlib.sha256(
            (spec + "\n" + scalar_text(c9, "audit_analysis_stack_sha256") +
             "\n" + recursive_sim_hash).encode()
        ).hexdigest()
        require(scalar_text(draws, "recursive_analysis_stack_sha256") ==
                recursive_analysis_hash,
                f"{label} recursive analysis-stack hash is inconsistent")

        dbar = np.asarray(draws["dbar"], dtype=float)
        usable = np.asarray(draws["recursive_usable_mask"], dtype=bool)
        excluded = np.asarray(draws["recursive_excluded_mask"], dtype=bool)
        exploded = np.asarray(draws["exploded"], dtype=bool)
        require(dbar.shape == usable.shape == excluded.shape == exploded.shape == (2000,),
                f"{label} recursive draw/mask shapes drifted")
        require(not np.any(np.isinf(dbar)),
                f"{label} recursive draw vector contains infinities")
        require_exact(usable, ~np.isnan(dbar), f"{label} recursive usable mask disagrees")
        require_exact(excluded, ~usable, f"{label} recursive masks are not complements")
        require_exact(draws["recursion_explosion_excluded"], exploded,
                      f"{label} explosion masks disagree")
        require(not np.any(exploded & usable), f"{label} retains recursion explosions")
        max_sig2 = np.asarray(draws["max_sig2"], dtype=float)
        mean_rv = np.asarray(draws["mean_rv"], dtype=float)
        sample_vars = np.asarray(draws["sample_vars_actual"], dtype=float)
        require(max_sig2.shape == mean_rv.shape == (2000,),
                f"{label} path diagnostic shapes drifted")
        require(sample_vars.shape == (6,) and np.all(np.isfinite(sample_vars)) and
                np.all(sample_vars >= 0),
                f"{label} actual sample-variance vector is invalid")
        require_near(scalar(draws, "sample_var_mean_actual"), sample_vars.mean(),
                     f"{label} archived sample-variance mean is inconsistent")
        require(np.all(np.isfinite(max_sig2)) and np.all(np.isfinite(mean_rv)),
                f"{label} path diagnostics contain non-finite values")
        attempted = np.asarray(draws["draw_had_six_start_rescue_attempt"], dtype=bool)
        retained = np.asarray(draws["draw_retained_after_six_start_rescue"], dtype=bool)
        asset_attempts = np.asarray(
            draws["n_asset_refits_with_six_start_rescue_attempt"], dtype=np.int64
        )
        asset_successes = np.asarray(
            draws["n_asset_refits_rescued_by_six_start"], dtype=np.int64
        )
        require(attempted.shape == retained.shape == asset_attempts.shape ==
                asset_successes.shape == (2000,), f"{label} rescue telemetry shapes drifted")
        require(np.all(asset_attempts >= 0) and np.all(asset_successes >= 0) and
                np.all(asset_successes <= asset_attempts),
                f"{label} rescue telemetry is invalid")
        require_exact(attempted, asset_attempts > 0,
                      f"{label} rescue flags disagree with asset attempts")
        require_exact(retained, attempted & usable,
                      f"{label} rescue-retention flags disagree with usable draws")

        floor_mask = np.asarray(draws["draw_had_variance_floor_hit"], dtype=bool)
        floor_hits = np.asarray(draws["n_variance_floor_hits"], dtype=np.int64)
        floor_paths = np.asarray(draws["n_asset_paths_with_variance_floor_hit"], dtype=np.int64)
        min_raw = np.asarray(draws["min_raw_variance_before_floor"], dtype=float)
        floor_by_asset = np.asarray(draws["variance_floor_hits_by_asset"], dtype=np.int64)
        min_raw_by_asset = np.asarray(draws["min_raw_variance_by_asset"], dtype=float)
        require(floor_mask.shape == floor_hits.shape == floor_paths.shape ==
                min_raw.shape == (2000,), f"{label} draw-level floor telemetry shapes drifted")
        require(floor_by_asset.shape == min_raw_by_asset.shape == (2000, 6),
                f"{label} asset-level floor telemetry shapes drifted")
        require_exact(draws["variance_floor_asset_names"], ASSETS,
                      f"{label} floor-telemetry asset order drifted")
        require(np.all(floor_hits >= 0) and np.all(floor_by_asset >= 0),
                f"{label} floor telemetry contains negative counts")
        require_exact(floor_mask, floor_hits > 0, f"{label} floor mask/counts disagree")
        require_exact(floor_hits, floor_by_asset.sum(axis=1),
                      f"{label} draw/asset floor-hit counts disagree")
        require_exact(floor_paths, (floor_by_asset > 0).sum(axis=1),
                      f"{label} floor path counts disagree")
        require(np.allclose(min_raw, min_raw_by_asset.min(axis=1),
                            rtol=0.0, atol=1e-12),
                f"{label} raw-variance minima disagree")

        values = dbar[usable]
        d_obs = float(scalar(draws, "d_bar_obs"))
        require(d_obs == float(scalar(c9, "d_bar_obs")),
                f"{label}/c9 observed statistic differs exactly")
        upper = int(np.sum(values >= d_obs))
        absolute = int(np.sum(np.abs(values) >= abs(d_obs)))
        lower = int(np.sum(values <= d_obs))
        refit_excluded = excluded & ~exploded
        recursive_ints = {
            "B": 2000, "B_requested": 2000, "B_used": usable.sum(),
            "n_total_exclusions": excluded.sum(),
            "n_recursion_explosion_exclusions": exploded.sum(),
            "n_refit_exclusions_after_six_start_rescue": refit_excluded.sum(),
            "n_draws_with_six_start_rescue_attempt": attempted.sum(),
            "n_draws_retained_after_six_start_rescue": retained.sum(),
            "n_draws_failed_after_six_start_rescue": np.sum(attempted & refit_excluded),
            "n_asset_refits_with_six_start_rescue_attempt": asset_attempts.sum(),
            "n_asset_refits_rescued_by_six_start": asset_successes.sum(),
            "n_upper_tail_hits_recursive": upper,
            "n_abs_stat_hits_recursive": absolute,
            "n_lower_tail_hits_recursive": lower,
            "n_draws_with_variance_floor_hit": floor_mask.sum(),
            "n_variance_floor_hits": floor_hits.sum(),
            "n_asset_paths_with_variance_floor_hit": floor_paths.sum(),
        }
        for field, expected in recursive_ints.items():
            _csv_int(row, field, expected, label)
        for field, expected in (
            ("fraction_total_excluded", excluded.mean()),
            ("fraction_recursion_explosion_excluded", exploded.mean()),
            ("fraction_refit_excluded_after_six_start_rescue", refit_excluded.mean()),
            ("d_bar_obs", d_obs),
            ("p_one_recursive", smoothed_tail(upper, len(values))),
            ("p_abs_recursive", smoothed_tail(absolute, len(values))),
            ("p_equal_tail_recursive", min(
                1.0,
                2 * min(smoothed_tail(upper, len(values)),
                        smoothed_tail(lower, len(values))),
            )),
            ("p_worst_low", smoothed_tail(upper, 2000)),
            ("p_worst_high", smoothed_tail(upper + int(excluded.sum()), 2000)),
            ("null_mean_recursive", values.mean()),
            ("null_sd_recursive", values.std()),
            ("fraction_draws_with_variance_floor_hit", floor_mask.mean()),
            ("min_raw_variance_before_floor", min_raw.min()),
            ("realized_var_median_recursive", np.median(mean_rv[usable])),
            ("sample_var_mean_actual", sample_vars.mean()),
        ):
            _csv_float(row, field, expected, label)
        require(float(row["runtime_s"]) > 0, f"{label} runtime must be positive")

        fixed_values = np.asarray(c9["null_t"], dtype=float)
        fixed_usable = np.asarray(c9["null_t_usable_mask"], dtype=bool)
        fixed_excluded = ~fixed_usable
        fixed_attempted = np.asarray(c9["draw_had_six_start_rescue_attempt_t"], dtype=bool)
        fixed_retained = np.asarray(c9["draw_retained_after_six_start_rescue_t"], dtype=bool)
        fixed_asset_attempts = np.asarray(
            c9["n_asset_refits_with_six_start_rescue_attempt_t"], dtype=np.int64
        )
        fixed_asset_successes = np.asarray(
            c9["n_asset_refits_rescued_by_six_start_t"], dtype=np.int64
        )
        f_upper = int(np.sum(fixed_values >= d_obs))
        f_absolute = int(np.sum(np.abs(fixed_values) >= abs(d_obs)))
        f_lower = int(np.sum(fixed_values <= d_obs))
        fixed_ints = {
            "B_used_fixed": fixed_usable.sum(), "B_requested_fixed": 2000,
            "n_total_exclusions_fixed": fixed_excluded.sum(),
            "n_draws_with_six_start_rescue_attempt_fixed": fixed_attempted.sum(),
            "n_draws_retained_after_six_start_rescue_fixed": fixed_retained.sum(),
            "n_draws_failed_after_six_start_rescue_fixed":
                np.sum(fixed_attempted & fixed_excluded),
            "n_asset_refits_with_six_start_rescue_attempt_fixed": fixed_asset_attempts.sum(),
            "n_asset_refits_rescued_by_six_start_fixed": fixed_asset_successes.sum(),
            "n_upper_tail_hits_fixed": f_upper, "n_abs_stat_hits_fixed": f_absolute,
            "n_lower_tail_hits_fixed": f_lower,
        }
        for field, expected in fixed_ints.items():
            _csv_int(row, field, expected, label)
        for field, expected in (
            ("fraction_total_excluded_fixed", fixed_excluded.mean()),
            ("p_one_fixed", smoothed_tail(f_upper, len(fixed_values))),
            ("p_abs_fixed", smoothed_tail(f_absolute, len(fixed_values))),
            ("p_equal_tail_fixed", min(
                1.0,
                2 * min(smoothed_tail(f_upper, len(fixed_values)),
                        smoothed_tail(f_lower, len(fixed_values))),
            )),
            ("null_mean_fixed", fixed_values.mean()), ("null_sd_fixed", fixed_values.std()),
        ):
            _csv_float(row, field, expected, label)

        floor_counts = floor_by_asset.sum(axis=0).astype(int)
        floor_draw_counts = (floor_by_asset > 0).sum(axis=0).astype(int)
        expected_floor_text = ";".join(f"{a}:{n}" for a, n in zip(ASSETS, floor_counts))
        expected_draw_text = ";".join(f"{a}:{n}" for a, n in zip(ASSETS, floor_draw_counts))
        require(str(row["variance_floor_hits_by_asset"]) == expected_floor_text,
                f"{label} CSV per-asset floor hits disagree")
        require(str(row["draws_with_variance_floor_hit_by_asset"]) == expected_draw_text,
                f"{label} CSV per-asset floor-draw counts disagree")
        expected_min_text = ";".join(
            f"{a}:{v:.16g}" for a, v in zip(ASSETS, min_raw_by_asset.min(axis=0))
        )
        require(str(row["min_raw_variance_by_asset"]) == expected_min_text,
                f"{label} CSV per-asset raw-variance minima disagree")
        for has_floor, suffix in ((True, "with_variance_floor_hit"),
                                  (False, "without_variance_floor_hit")):
            mask = usable & (floor_mask if has_floor else ~floor_mask)
            subset = dbar[mask]
            _csv_int(row, f"n_usable_draws_{suffix}", len(subset), label)
            _csv_int(row, f"n_upper_tail_hits_recursive_{suffix}",
                     np.sum(subset >= d_obs), label)
            _csv_int(row, f"n_abs_stat_hits_recursive_{suffix}",
                     np.sum(np.abs(subset) >= abs(d_obs)), label)
            if len(subset):
                _csv_float(row, f"null_mean_recursive_{suffix}", subset.mean(), label)
                _csv_float(row, f"null_sd_recursive_{suffix}", subset.std(), label)
            else:
                require(pd.isna(row[f"null_mean_recursive_{suffix}"]) and
                        pd.isna(row[f"null_sd_recursive_{suffix}"]),
                        f"{label} empty floor subset moments must be NaN")

        for csv_key, expected in (
            ("design_sha256", scalar_text(c9, "audit_design_sha256")),
            ("fixed_null_dgp_sha256", scalar_text(c9, "audit_fixed_null_dgp_sha256")),
            ("refitter_sha256", scalar_text(c9, "audit_refitter_sha256")),
            ("simulator_sha256", scalar_text(c9, "audit_simulator_sha256")),
            ("analysis_stack_sha256", scalar_text(c9, "audit_analysis_stack_sha256")),
            ("recursive_simulator_sha256", recursive_sim_hash),
            ("recursive_analysis_stack_sha256", recursive_analysis_hash),
            ("design_sha256_fixed", scalar_text(c9, "audit_design_sha256")),
            ("fixed_null_dgp_sha256_fixed", scalar_text(c9, "audit_fixed_null_dgp_sha256")),
            ("refitter_sha256_fixed", scalar_text(c9, "audit_refitter_sha256")),
            ("simulator_sha256_fixed", scalar_text(c9, "audit_simulator_sha256")),
            ("analysis_stack_sha256_fixed", scalar_text(c9, "audit_analysis_stack_sha256")),
        ):
            require(str(row[csv_key]) == expected, f"{label} CSV {csv_key} differs")
        require(bool(row["fixed_audit_locally_reconstructed_equal"]),
                f"{label} CSV lost fixed-audit equality certification")
        for field, expected in (
            ("base_seed", FIT_SEED), ("recursive_seed_schema_version", 1),
            ("recursive_draw_seed_offset", DRAW_SEED_OFFSET),
            ("first_draw_seed", expected_seeds[0]),
            ("last_draw_seed", expected_seeds[-1]), ("n_calendar", 2434),
        ):
            _csv_int(row, field, expected, label)
        require(str(row["availability_pattern_codes"]) == "55;63" and
                str(row["availability_pattern_counts"]) == "243;2191",
                f"{label} CSV union-calendar summary drifted")
        for field in ("p_one_recursive", "p_abs_recursive", "p_equal_tail_recursive",
                      "p_worst_low", "p_worst_high", "p_one_fixed", "p_abs_fixed",
                      "p_equal_tail_fixed"):
            require(0.0 <= float(row[field]) <= 1.0,
                    f"{label} CSV {field} leaves the probability interval")
        outputs[spec] = (row, draws)

    require(sha256_file(RESULTS / "c19-recursive-sensitivity-baseline.csv") ==
            sha256_file(RESULTS / "c19-recursive-sensitivity.csv"),
            "generic c19 CSV is not the byte-identical baseline alias")
    require(sha256_file(RESULTS / "c19-recursive-draws-baseline.npz") ==
            sha256_file(RESULTS / "c19-recursive-draws.npz"),
            "generic c19 NPZ is not the byte-identical baseline alias")
    finding_snippets = []
    for spec in ("baseline", "crisis"):
        row = outputs[spec][0]
        finding_snippets.append((
            f"{spec} comparison row",
            f"| {spec} | {int(row['B_used_fixed'])} / {int(row['B_used'])} | "
            f"{float(row['p_one_fixed']):.4f} / "
            f"{float(row['p_one_recursive']):.4f} | "
            f"{float(row['p_abs_fixed']):.4f} / "
            f"{float(row['p_abs_recursive']):.4f} | "
            f"{float(row['null_mean_fixed']):.4f} / "
            f"{float(row['null_mean_recursive']):.4f} | "
            f"{float(row['null_sd_fixed']):.4f} / "
            f"{float(row['null_sd_recursive']):.4f} | "
            f"{float(row['fraction_draws_with_variance_floor_hit']):.1%} |",
        ))
    require_finding(
        RESULTS / "c19-recursive-sensitivity-FINDING.md",
        "# C19 -- Fixed-path versus floored recursive sensitivity",
        finding_snippets,
        "c19",
    )
    return outputs


def validate_c20(c9_baseline, c9_rows):
    label = "c20"
    csv_path = RESULTS / "c20-nuc-sensitivity.csv"
    require(csv_path.is_file(), "required c20 results CSV is missing")
    frame = pd.read_csv(csv_path, float_precision="round_trip")
    draws = load_npz(RESULTS / "c20-nuc-sensitivity-draws.npz")
    required = {
        "audit_schema_version", "seed_schema_version", "seed_schema", "spec",
        "nu_c_roles", "nu_c_provenance", "nu_c_grid", "fitted_nu_null_by_asset",
        "fitted_nu_c_median", "fit_seed", "draw_seed_offset", "first_draw_seed",
        "last_draw_seed", "B_requested", "draw_seeds", "d_bar_obs", "multiplier",
        "rho_return", "rho_resid", "null_t_full", "null_t_usable_mask",
        "null_t_excluded_mask", "draw_had_six_start_rescue_attempt",
        "draw_retained_after_six_start_rescue",
        "n_asset_refits_with_six_start_rescue_attempt",
        "n_asset_refits_rescued_by_six_start", "paired_usable_with_baseline_mask",
        "paired_difference_vs_baseline", "B_used", "n_dropped",
        "n_upper_tail_hits", "n_abs_stat_hits", "n_lower_tail_hits", "p_one",
        "p_abs", "p_equal_tail", "null_mean", "null_sd",
        "n_paired_usable_with_baseline", "paired_diff_mean", "paired_diff_sd",
        "paired_diff_max_abs", *HASH_KEYS, "source_c9_baseline_npz",
        "source_c9_baseline_npz_sha256", "source_c9_results_csv",
        "source_c9_results_csv_sha256", "source_c9_code_sha256",
        "baseline_exact_match_c9",
        "c20_analysis_contract_version", "c20_analysis_contract",
        "c20_analysis_sha256", "c20_sensitivity_laws_sha256",
        "c20_source_sha256", "c20_parent_c9_analysis_stack_sha256",
        "c20_downstream_analysis_stack_sha256",
    }
    require_fields(draws, required, label)
    require((int(scalar(draws, "audit_schema_version")),
             int(scalar(draws, "seed_schema_version"))) == (1, 2),
            "c20 audit/seed schema versions drifted")
    require(scalar_text(draws, "spec") == "baseline",
            "c20 archive is not the baseline specification")
    require((int(scalar(draws, "fit_seed")), int(scalar(draws, "draw_seed_offset")),
             int(scalar(draws, "B_requested"))) ==
            (FIT_SEED, DRAW_SEED_OFFSET, 2000), "c20 seed/request contract drifted")
    expected_seeds = np.arange(FIT_SEED + DRAW_SEED_OFFSET,
                               FIT_SEED + DRAW_SEED_OFFSET + 2000,
                               dtype=np.int64)
    require_exact(draws["draw_seeds"], expected_seeds, "c20 draw-seed array drifted")
    require((int(scalar(draws, "first_draw_seed")),
             int(scalar(draws, "last_draw_seed"))) ==
            (int(expected_seeds[0]), int(expected_seeds[-1])),
            "c20 first/last draw-seed fields disagree")
    roles = np.asarray(draws["nu_c_roles"]).astype(str)
    provenance = np.asarray(draws["nu_c_provenance"]).astype(str)
    nu_grid = np.asarray(draws["nu_c_grid"], dtype=float)
    require_exact(roles, np.asarray(["fitted_median_baseline",
                                     "alternative_sensitivity",
                                     "alternative_sensitivity"]),
                  "c20 copula-df roles/order drifted")
    require_exact(provenance, np.asarray([
        "median of the six null-fit marginal nu values",
        "alternative sensitivity value; not a rank-based estimate",
        "alternative lighter-tail sensitivity value",
    ]), "c20 copula-df provenance labels drifted")
    fitted_nu = np.asarray(draws["fitted_nu_null_by_asset"], dtype=float)
    fitted_median = float(scalar(draws, "fitted_nu_c_median"))
    require_exact(fitted_nu, c9_baseline["audit_nu_null_by_asset"],
                  "c20/c9 fitted marginal nu vector differs")
    require(fitted_median == float(scalar(c9_baseline, "audit_nu_c_null")) ==
            float(np.median(fitted_nu)),
            "c20 fitted median copula df differs from c9")
    require_exact(nu_grid, np.asarray([fitted_median, 5.9, 8.0]),
                  "c20 copula-df grid drifted")
    require(len(np.unique(nu_grid)) == 3, "c20 copula-df rows are not uniquely indexable")
    c20_provenance = c20.current_c20_analysis_provenance(
        tuple(roles), tuple(provenance), tuple(nu_grid),
        scalar_text(c9_baseline, "audit_analysis_stack_sha256"),
    )
    require_current_provenance(draws, c20_provenance, "c20 downstream provenance")

    full = np.asarray(draws["null_t_full"], dtype=float)
    usable = np.asarray(draws["null_t_usable_mask"], dtype=bool)
    excluded = np.asarray(draws["null_t_excluded_mask"], dtype=bool)
    attempted = np.asarray(draws["draw_had_six_start_rescue_attempt"], dtype=bool)
    retained = np.asarray(draws["draw_retained_after_six_start_rescue"], dtype=bool)
    asset_attempts = np.asarray(
        draws["n_asset_refits_with_six_start_rescue_attempt"], dtype=np.int64
    )
    asset_successes = np.asarray(
        draws["n_asset_refits_rescued_by_six_start"], dtype=np.int64
    )
    paired_mask = np.asarray(draws["paired_usable_with_baseline_mask"], dtype=bool)
    paired_diff = np.asarray(draws["paired_difference_vs_baseline"], dtype=float)
    for name, array in (("full draws", full), ("usable masks", usable),
                        ("excluded masks", excluded), ("rescue attempts", attempted),
                        ("rescue retention", retained), ("asset attempts", asset_attempts),
                        ("asset successes", asset_successes), ("paired masks", paired_mask),
                        ("paired differences", paired_diff)):
        require(array.shape == (3, 2000), f"c20 {name} shape drifted")
    require_exact(usable, ~np.isnan(full), "c20 usable masks do not identify non-NaN draws")
    require(not np.any(np.isinf(full)), "c20 full draw matrix contains infinities")
    require_exact(excluded, ~usable, "c20 masks are not complements")
    require(np.all(asset_attempts >= 0) and np.all(asset_successes >= 0) and
            np.all(asset_successes <= asset_attempts), "c20 rescue telemetry is invalid")
    require_exact(attempted, asset_attempts > 0,
                  "c20 rescue flags disagree with asset attempts")
    require_exact(retained, attempted & usable,
                  "c20 rescue-retention flags disagree with usable draws")
    expected_paired = usable & usable[0][None, :]
    require_exact(paired_mask, expected_paired, "c20 baseline-pairing masks are inconsistent")
    expected_diff = np.full((3, 2000), np.nan)
    expected_diff[expected_paired] = (full - full[0])[expected_paired]
    require_exact(paired_diff, expected_diff, "c20 paired differences are inconsistent")

    # The first c20 row is an exact rerun of c9, including all diagnostics.
    for actual, expected, name in (
        (full[0], c9_baseline["null_t_full"], "full statistic vector"),
        (usable[0], c9_baseline["null_t_usable_mask"], "usable mask"),
        (excluded[0], c9_baseline["null_t_excluded_mask"], "excluded mask"),
        (attempted[0], c9_baseline["draw_had_six_start_rescue_attempt_t"],
         "rescue-attempt mask"),
        (retained[0], c9_baseline["draw_retained_after_six_start_rescue_t"],
         "rescue-retention mask"),
        (asset_attempts[0], c9_baseline["n_asset_refits_with_six_start_rescue_attempt_t"],
         "asset rescue attempts"),
        (asset_successes[0], c9_baseline["n_asset_refits_rescued_by_six_start_t"],
         "asset rescue successes"),
    ):
        require_exact(actual, expected, f"c20 baseline {name} differs from c9")
    require(bool(scalar(draws, "baseline_exact_match_c9")),
            "c20 lost its exact c9 baseline-match certification")
    for field in ("d_bar_obs", "multiplier", "rho_return", "rho_resid"):
        require(float(scalar(draws, field)) == float(c9_rows.loc["baseline", field]),
                f"c20/c9 baseline {field} differs exactly")

    # Recompute all row summaries from the full seed-indexed vectors.
    d_obs = float(scalar(draws, "d_bar_obs"))
    summary_int_fields = ("B_used", "n_dropped", "n_upper_tail_hits",
                          "n_abs_stat_hits", "n_lower_tail_hits",
                          "n_paired_usable_with_baseline")
    summary_float_fields = ("p_one", "p_abs", "p_equal_tail", "null_mean",
                            "null_sd", "paired_diff_mean", "paired_diff_sd",
                            "paired_diff_max_abs")
    recomputed = []
    for i in range(3):
        values = full[i, usable[i]]
        upper = int(np.sum(values >= d_obs))
        absolute = int(np.sum(np.abs(values) >= abs(d_obs)))
        lower = int(np.sum(values <= d_obs))
        paired_values = paired_diff[i, paired_mask[i]]
        recomputed.append({
            "B_used": len(values), "n_dropped": int(excluded[i].sum()),
            "n_upper_tail_hits": upper, "n_abs_stat_hits": absolute,
            "n_lower_tail_hits": lower,
            "p_one": smoothed_tail(upper, len(values)),
            "p_abs": smoothed_tail(absolute, len(values)),
            "p_equal_tail": min(1.0, 2 * min(smoothed_tail(upper, len(values)),
                                                   smoothed_tail(lower, len(values)))),
            "null_mean": values.mean(), "null_sd": values.std(),
            "n_paired_usable_with_baseline": len(paired_values),
            "paired_diff_mean": paired_values.mean(),
            "paired_diff_sd": paired_values.std(),
            "paired_diff_max_abs": np.max(np.abs(paired_values)),
        })
        for field in summary_int_fields:
            require(int(np.asarray(draws[field])[i]) == int(recomputed[-1][field]),
                    f"c20 row {i} archive {field} is inconsistent")
        for field in summary_float_fields:
            require_near(np.asarray(draws[field])[i], recomputed[-1][field],
                         f"c20 row {i} archive {field} is inconsistent")
        for field in ("p_one", "p_abs", "p_equal_tail"):
            require(0.0 <= recomputed[-1][field] <= 1.0,
                    f"c20 row {i} recomputed {field} leaves the probability interval")

    design_hash = scalar_text(c9_baseline, "audit_design_sha256")
    refitter_hash = scalar_text(c9_baseline, "audit_refitter_sha256")
    simulator_hash = scalar_text(c9_baseline, "audit_simulator_sha256")
    expected_hashes = []
    for i, nu_c in enumerate(nu_grid):
        dgp_hash = hash_named_arrays([
            ("asset_names", c9_baseline["audit_asset_names"]),
            ("null_params", c9_baseline["audit_null_params_by_asset"]),
            ("mean_returns", c9_baseline["audit_null_mean_returns"]),
            ("sigma2_null_concat", c9_baseline["audit_null_sigma2_concat"]),
            ("sigma2_null_offsets", c9_baseline["audit_null_sigma2_offsets"]),
            ("common_pos_concat", c9_baseline["audit_common_pos_concat"]),
            ("common_pos_offsets", c9_baseline["audit_common_pos_offsets"]),
            ("simulation_calendar_ns", c9_baseline["audit_simulation_calendar_ns"]),
            ("calendar_pos_concat", c9_baseline["audit_calendar_pos_concat"]),
            ("calendar_pos_offsets", c9_baseline["audit_calendar_pos_offsets"]),
            ("nu_null", c9_baseline["audit_nu_null_by_asset"]),
            ("nu_c_null", np.asarray(nu_c, dtype=np.float64)),
            ("R_z", c9_baseline["audit_R_z"]), ("L_z", c9_baseline["audit_L_z"]),
            ("use_t_copula", c9_baseline["audit_use_t_copula"]),
        ])
        analysis_hash = hashlib.sha256(
            ("baseline\n" + design_hash + "\n" + dgp_hash + "\n" +
             refitter_hash + "\n" + simulator_hash).encode()
        ).hexdigest()
        expected_hashes.append((design_hash, dgp_hash, refitter_hash,
                                simulator_hash, analysis_hash))
        for archive_key, expected in zip(HASH_KEYS, expected_hashes[-1]):
            require(str(np.asarray(draws[archive_key])[i]) == expected,
                    f"c20 row {i} {archive_key} is inconsistent")

    require(scalar_text(draws, "source_c9_baseline_npz") ==
            "c9-tcopula-draws-baseline.npz",
            "c20 source c9 archive label drifted")
    require(scalar_text(draws, "source_c9_baseline_npz_sha256") ==
            sha256_file(RESULTS / "c9-tcopula-draws-baseline.npz"),
            "c20 source c9 archive SHA-256 is stale")
    require(scalar_text(draws, "source_c9_results_csv") == "c9-tcopula-results.csv",
            "c20 source c9 results CSV label drifted")
    source_c9_results_sha256 = sha256_file(RESULTS / "c9-tcopula-results.csv")
    require(scalar_text(draws, "source_c9_results_csv_sha256") ==
            source_c9_results_sha256,
            "c20 source c9 results CSV SHA-256 is stale")
    # C20 records the whole C9 source file as it existed when C20 ran. The
    # current file differs only in write_finding(), whose human-facing legacy-
    # key disclosure was repaired after the numerical run. Above,
    # _validate_c9_audit independently requires the archived refitter and
    # simulator contracts to match the current executable analysis functions.
    require(scalar_text(draws, "source_c9_code_sha256") ==
            C20_C9_SOURCE_AT_RUN_SHA256,
            "c20 no longer records the exact c9 whole-source hash at run time")
    require(sha256_file(HERE / "c9_tcopula_bootstrap.py") ==
            C9_CURRENT_SOURCE_SHA256,
            "current c9 source differs from the audited post-run renderer repair")

    required_csv = {
        "spec", "nu_c_role", "nu_c", "nu_c_provenance", "fit_seed",
        "draw_seed_offset", "first_draw_seed", "last_draw_seed", "B_requested",
        *summary_int_fields, *summary_float_fields, "d_bar_obs", "multiplier",
        "frac_dropped", "n_draws_with_six_start_rescue_attempt",
        "n_draws_retained_after_six_start_rescue",
        "n_draws_failed_after_six_start_rescue",
        "n_asset_refits_with_six_start_rescue_attempt",
        "n_asset_refits_rescued_by_six_start",
        "rho_return", "rho_resid", "design_sha256", "fixed_null_dgp_sha256",
        "refitter_sha256", "simulator_sha256", "analysis_stack_sha256",
        "source_c9_baseline_npz_sha256", "source_c9_results_csv_sha256",
        "c9_baseline_comparison",
        "c20_analysis_contract_version", "c20_analysis_sha256",
        "c20_sensitivity_laws_sha256", "c20_source_sha256",
        "c20_parent_c9_analysis_stack_sha256",
        "c20_downstream_analysis_stack_sha256",
    }
    require_fields(frame.columns, required_csv, "c20 results CSV")
    require(len(frame) == 3, "c20 results CSV must contain exactly three rows")
    require(frame["nu_c"].is_unique, "c20 results CSV copula-df rows are duplicated")
    require_exact(frame["nu_c_role"].to_numpy(dtype=str), roles,
                  "c20 CSV role order differs from NPZ")
    require_exact(frame["nu_c_provenance"].to_numpy(dtype=str), provenance,
                  "c20 CSV provenance order differs from NPZ")
    require_exact(frame["nu_c"].to_numpy(dtype=float), nu_grid,
                  "c20 CSV copula-df grid differs from NPZ")
    for i, row in frame.iterrows():
        require(str(row["spec"]) == "baseline", f"c20 CSV row {i} spec drifted")
        for field, expected in (
            ("fit_seed", FIT_SEED), ("draw_seed_offset", DRAW_SEED_OFFSET),
            ("first_draw_seed", expected_seeds[0]),
            ("last_draw_seed", expected_seeds[-1]), ("B_requested", 2000),
        ):
            _csv_int(row, field, expected, f"c20 row {i}")
        for field in summary_int_fields:
            _csv_int(row, field, recomputed[i][field], f"c20 row {i}")
        for field in summary_float_fields:
            _csv_float(row, field, recomputed[i][field], f"c20 row {i}")
        _csv_float(row, "frac_dropped", excluded[i].mean(), f"c20 row {i}")
        for field, expected in (
            ("n_draws_with_six_start_rescue_attempt", attempted[i].sum()),
            ("n_draws_retained_after_six_start_rescue", retained[i].sum()),
            ("n_draws_failed_after_six_start_rescue",
             np.sum(attempted[i] & excluded[i])),
            ("n_asset_refits_with_six_start_rescue_attempt", asset_attempts[i].sum()),
            ("n_asset_refits_rescued_by_six_start", asset_successes[i].sum()),
        ):
            _csv_int(row, field, expected, f"c20 row {i}")
        for field in ("d_bar_obs", "multiplier", "rho_return", "rho_resid"):
            _csv_float(row, field, scalar(draws, field), f"c20 row {i}")
        for csv_key, expected in zip(
            ("design_sha256", "fixed_null_dgp_sha256", "refitter_sha256",
             "simulator_sha256", "analysis_stack_sha256"), expected_hashes[i]
        ):
            require(str(row[csv_key]) == expected, f"c20 row {i} CSV {csv_key} differs")
        require(str(row["source_c9_baseline_npz_sha256"]) ==
                scalar_text(draws, "source_c9_baseline_npz_sha256"),
                f"c20 row {i} source archive hash differs")
        require(str(row["source_c9_results_csv_sha256"]) == source_c9_results_sha256,
                f"c20 row {i} source c9 results CSV hash differs")
        expected_comparison = "exact_match" if i == 0 else "not_applicable"
        require(str(row["c9_baseline_comparison"]) == expected_comparison,
                f"c20 row {i} c9 comparison label drifted")
        for key, expected in c20_provenance.items():
            if key == "c20_analysis_contract":
                continue
            require(str(row[key]) == str(expected),
                    f"c20 row {i} CSV {key} differs from current provenance")
    finding_snippets = []
    for i, row in frame.iterrows():
        finding_snippets.append((
            f"row {i} sensitivity result",
            f"| {row['nu_c_role']} | {float(row['nu_c']):.6g} | "
            f"{int(row['B_used'])}/{int(row['B_requested'])} | "
            f"{float(row['p_one']):.4f} | {float(row['p_abs']):.4f} | "
            f"{float(row['null_mean']):.4f} | {float(row['null_sd']):.4f} | "
            f"{int(row['n_paired_usable_with_baseline'])} | "
            f"{float(row['paired_diff_mean']):+.4f} |",
        ))
    finding_snippets.append((
        "source c9 archive hash",
        f"- c9 baseline NPZ SHA-256: {scalar_text(draws, 'source_c9_baseline_npz_sha256')}",
    ))
    finding_snippets.append((
        "source c9 results hash",
        f"- c9 results CSV SHA-256: {source_c9_results_sha256}",
    ))
    for label_text, key in (
        ("analysis hash", "c20_analysis_sha256"),
        ("sensitivity-laws hash", "c20_sensitivity_laws_sha256"),
        ("source hash", "c20_source_sha256"),
        ("downstream analysis-stack hash", "c20_downstream_analysis_stack_sha256"),
    ):
        finding_snippets.append((
            f"c20 {label_text}",
            f"- c20 {label_text}: {c20_provenance[key]}",
        ))
    require_finding(
        RESULTS / "c20-nuc-sensitivity-FINDING.md",
        "# C20 -- Paired copula-df sensitivity",
        finding_snippets,
        "c20",
    )
    return frame, draws


def validate_required_release_surface():
    required = [
        ROOT / "requirements-lock.txt",
        RESULTS / "c3-bai-perron-results.csv",
        RESULTS / "c3-subsample-persistence.csv",
        RESULTS / "c3-break-summary.md",
        RESULTS / "c8e-persistence-se.csv",
        RESULTS / "c9-tcopula-results.csv",
        RESULTS / "c9-tcopula-bootstrap-FINDING.md",
        *(RESULTS / f"c9-tcopula-draws-{spec}.npz" for spec in SPECS),
        RESULTS / "c8h-break-ccc-bootstrap-results.csv",
        RESULTS / "c8h-break-controls-ccc-FINDING.md",
        RESULTS / "c8h-break-ccc-draws-full.npz",
        RESULTS / "c8h-break-ccc-draws-crisis.npz",
        RESULTS / "c10-size-study-results.csv",
        RESULTS / "c10-size-study-metadata.csv",
        RESULTS / "c10-size-study-draws.npz",
        RESULTS / "c10-size-study-FINDING.md",
        RESULTS / "c10b-dgp-perturbation-results.csv",
        RESULTS / "c10b-dgp-perturbation-draws.npz",
        RESULTS / "c10b-dgp-perturbation-FINDING.md",
        RESULTS / "c13-rung4-recompute-results.csv",
        RESULTS / "c13-rung4-perasset-draws.npz",
        RESULTS / "c13-rung4-recompute-FINDING.md",
        RESULTS / "c19-recursive-sensitivity-baseline.csv",
        RESULTS / "c19-recursive-sensitivity-crisis.csv",
        RESULTS / "c19-recursive-sensitivity.csv",
        RESULTS / "c19-recursive-draws-baseline.npz",
        RESULTS / "c19-recursive-draws-crisis.npz",
        RESULTS / "c19-recursive-draws.npz",
        RESULTS / "c19-recursive-sensitivity-FINDING.md",
        RESULTS / "c20-nuc-sensitivity.csv",
        RESULTS / "c20-nuc-sensitivity-draws.npz",
        RESULTS / "c20-nuc-sensitivity-FINDING.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    require(not missing, f"required final release artifacts are missing: {missing}")
    temporary_tokens = (".write-tmp", ".telemetry-tmp", ".sync-fixed-tmp")
    leftovers = []
    for directory in (ROOT, HERE, RESULTS):
        for path in directory.iterdir():
            if path.is_file() and any(token in path.name for token in temporary_tokens):
                leftovers.append(str(path.relative_to(ROOT)))
    require(not leftovers, f"temporary/incomplete producer artifacts remain: {sorted(set(leftovers))}")


def main():
    validate_required_release_surface()
    validate_c3_c8e()
    events = pd.read_csv(DATA / "events.csv")
    census = pd.read_csv(RESULTS / "c1-dropout-census.csv")
    corrected_events = {
        "QuadrigaCX exchange collapses": "2019-01-31",
        "China mining crackdown announcement": "2021-05-21",
        "Celsius freezes withdrawals; credit crisis unfolds": "2022-06-13",
        "EU Parliament final passage of MiCA": "2023-04-20",
        "XRP litigation concludes; fine/injunction terms finalized; appeals withdrawn": "2025-08-07",
    }
    corrected_census = {
        "QuadrigaCX": "2019-01-31",
        "China mining ban": "2021-05-21",
        "Celsius freeze": "2022-06-13",
        "EU MiCA passed": "2023-04-20",
        "XRP case ends": "2025-08-07",
    }
    for title, date in corrected_events.items():
        actual = events.loc[events["title"] == title, "date"].tolist()
        require(actual == [date], f"curated event date drifted: {title}")
    for label, date in corrected_census.items():
        actual = census.loc[census["label"] == label, "date"].tolist()
        require(actual == [date], f"candidate-census date drifted: {label}")
    legacy = json.loads((DATA / "events_reclassified.json").read_text())
    legacy_celsius = [e["date"] for e in legacy["events"] if "celsius" in e["title"].lower()]
    require(legacy_celsius == ["2022-06-12"],
            "legacy smoke-test snapshot no longer preserves its historical Celsius date")

    require(len(census) == 135, "candidate-pool size drifted")
    by_cat = census.groupby("tentative_category").size().to_dict()
    require(by_cat == {"Infrastructure": 82, "Regulatory": 53},
            "candidate-pool category counts drifted")
    pass_counts = census.groupby("tentative_category")["stage2_std_pass"].sum().astype(int).to_dict()
    require(pass_counts == {"Infrastructure": 53, "Regulatory": 41},
            "mechanical-screen pass counts drifted")

    returns_all = pd.read_csv(RESULTS / "c-gate-returns-unified-results.csv")
    returns = returns_all.iloc[0]
    require((int(returns["asset_event_cells_total"]),
             int(returns["asset_event_cells_usable"]),
             int(returns["missing_estimation_cells"]),
             int(returns["incomplete_event_window_cells"]),
             int(returns["other_error_cells"])) == (300, 293, 6, 1, 0),
            "first-moment coverage accounting drifted")
    require(returns["missing_asset_event_ids"] ==
            "1:BNB;2:BNB;3:BNB;4:BNB;5:BNB;6:BNB;50:BNB",
            "first-moment omitted-cell identities drifted")
    require(near(returns["diff_pp"], 8.693161895776171), "first-moment difference drifted")
    require(near(returns["block_p_two"], 0.202), "block-bootstrap p drifted")
    require(near(returns["im_p"], 0.21643388452547363), "event-level Welch p drifted")
    require((int(returns["legacy_negative_exact_permutation_assignments"]),
             int(returns["legacy_negative_exact_permutation_extreme"])) == (6435, 5961) and
            near(returns["legacy_negative_exact_permutation_p_two"], 0.9263403263403264),
            "legacy exact-permutation result drifted")
    require(near(returns["legacy_negative_market_diff_pp"], 5.884266731036974) and
            near(returns["legacy_negative_market_block_p_two"], 0.5236) and
            near(returns["legacy_negative_market_welch_p"], 0.5834889367516949) and
            near(returns["legacy_negative_market_permutation_p_two"], 0.5743589743589743),
            "event-equal market-model robustness drifted")
    require(sha256_file(RESULTS / "c-gate-returns-unified-results.csv") ==
            sha256_file(RESULTS / "c11-returns-results.csv"),
            "c11 release alias is not byte-identical")

    c2 = pd.read_csv(RESULTS / "c2-summary-table.csv").set_index("spec")
    expected_c2 = {
        "S1_baseline": (26, 24, 2.0443214810809556, 0.5863028473346455,
                        3.48680060234564, 0.005918),
        "S2_relaxed": (69, 46, 1.470941, 1.169860, 1.257365, 0.592594),
        "S3_nofilter": (82, 53, 0.260021, 0.534459, 0.486513, 0.287571),
        "S4_strict": (42, 36, 3.991057, 3.009557, 1.326128, 0.384266),
    }
    for spec, (n_i, n_r, d_i, d_r, mult, p) in expected_c2.items():
        row = c2.loc[spec]
        require((int(row["n_infra_events"]), int(row["n_reg_events"])) == (n_i, n_r),
                f"c2 {spec} event counts drifted")
        require(near(row["mean_infra_coef"], d_i) and near(row["mean_reg_coef"], d_r),
                f"c2 {spec} coefficient means drifted")
        require(near(row["multiplier"], mult) and near(row["welch_p"], p),
                f"c2 {spec} summary drifted")

    c9_rows, c9_payloads = validate_c9()
    validate_c8h(c9_rows, c9_payloads)
    _c10_rows, _c10_meta, c10_draws = validate_c10(
        c9_payloads["baseline"], c9_rows.loc["baseline"]
    )
    validate_c10b(c9_payloads["baseline"], c10_draws)
    validate_c13(c9_payloads["baseline"], c10_draws)
    validate_c19(c9_payloads)
    validate_c20(c9_payloads["baseline"], c9_rows)

    two = pd.read_csv(RESULTS / "c2b-two-asset-result.csv").iloc[0]
    require((int(two["n_infra_events"]), int(two["n_reg_events"])) == (53, 41),
            "mechanical two-asset event counts drifted")
    require(1.0 < float(two["multiplier"]) < 2.0 and float(two["welch_p"]) > 0.10,
            "mechanical two-asset scope-condition verdict changed")

    c14 = pd.read_csv(RESULTS / "c14-garch-diagnostics-per-asset.csv")
    require(c14["converged"].all(), "one or more c14 baseline fits did not converge")
    require(int(c14["variance_floor_hits"].sum()) == 0,
            "the numerical variance floor binds in a reported baseline fit")
    require((c14["alpha"] + c14["gamma"] >= 0).all(),
            "alpha + gamma positivity condition fails in a reported baseline fit")

    c18 = pd.read_csv(RESULTS / "c18-model-comparison-refit.csv")
    require(c18["converged"].all() and (c18["n_starts_converged"] == c18["n_starts"]).all(),
            "one or more c18 controlled fits did not converge")
    expected_k = {"GARCH(1,1)": 5, "GJR-GARCH": 6, "GJR-GARCH-X": 11}
    require(all(int(r.k) == expected_k[r.model] for r in c18.itertuples()),
            "c18 primary parameter counts are not 5/6/11")
    require((c18["AIC"] - c18["AIC_kmu"]).abs().max() < 1e-10,
            "c18 primary AIC is not the mean-counted convention")
    expected_aic_winners = {
        "btc": "GJR-GARCH-X", "eth": "GJR-GARCH-X", "xrp": "GJR-GARCH-X",
        "bnb": "GARCH(1,1)", "ltc": "GJR-GARCH-X", "ada": "GJR-GARCH-X",
    }
    for asset, model in expected_aic_winners.items():
        sub = c18[c18["asset"] == asset]
        require(sub.loc[sub["AIC"].idxmin(), "model"] == model,
                f"c18 AIC winner drifted for {asset}")
        require(sub.loc[sub["BIC"].idxmin(), "model"] == "GARCH(1,1)",
                f"c18 BIC winner drifted for {asset}")
    bnb = c18[c18["asset"] == "bnb"].set_index("model")
    bnb_gap = bnb.loc["GJR-GARCH-X", "AIC"] - bnb.loc["GARCH(1,1)", "AIC"]
    require(near(bnb_gap, 0.664579184624, tol=5e-5), "BNB AIC margin drifted")

    c17 = pd.read_csv(RESULTS / "c17-model-comparison.csv")
    joined = c17.merge(c18, left_on=["asset", "model"], right_on=[c18["asset"].str.upper(), "model"])
    require(len(joined) == 18, "c17 compact model table is incomplete")
    require((joined["k_params"] == joined["k"]).all(), "c17/c18 parameter counts disagree")
    require((joined["AIC_x"] - joined["AIC_y"]).abs().max() <= 5.01e-4,
            "c17/c18 AIC values disagree beyond CSV rounding")

    warmup = pd.read_csv(RESULTS / "c12-granger-rigour-missingness.csv").set_index("asset")
    expected_imputed = {"btc": 24, "eth": 24, "xrp": 24, "bnb": 0, "ltc": 24, "ada": 24}
    require(warmup["n_imputed"].astype(int).to_dict() == expected_imputed,
            "c12 normalized-series warm-up counts drifted")
    estimable = warmup.drop(index="bnb")
    require((estimable["logit_p"] > 0.10).all(),
            "c12 warm-up logit diagnostic changed its reported verdict")
    require(math.isnan(float(warmup.loc["bnb", "logit_p"])) and
            math.isnan(float(warmup.loc["bnb", "chi2_p"])),
            "c12 BNB warm-up diagnostic should be non-estimable")
    chi_flags = set(estimable.index[estimable["chi2_p"] < 0.05])
    require(chi_flags == {"ltc"} and near(estimable.loc["ltc", "chi2_p"], 0.005935, tol=5e-7),
            "c12 secondary chi-square warm-up flag drifted")

    expected_scripts = [
        "c1_build_candidate_pool.py", "c2_relaxed_threshold_sensitivity.py",
        "c2b_two_asset_point.py", "c2c_corrected_figure.py",
        "c3_bai_perron.py", "c4_granger_causality.py", "c5_pseudoreplication_test.py",
        "c6_garchx_clustered.py", "c7_ccc_garchx_bootstrap.py",
        "c8a_break_controls.py", "c8b_anticipation_windows.py",
        "c8c_mechanical_rolling_winsor.py", "c8d_constant_mean.py",
        "c8e_persistence_se.py", "c8f_weekly_granger_fdr.py",
        "c8h_break_controls_ccc_bootstrap.py", "c9_tcopula_bootstrap.py",
        "c10_size_study.py", "c10b_dgp_perturbation.py",
        "c11_returns_block_bootstrap.py", "c12_granger_rigour.py",
        "c13_rung4_recompute.py", "c14_garch_diagnostics.py",
        "c15_granger_event_dummies.py", "c16_nonoverlap_pre_dummy.py",
        "c17_model_comparison.py", "c18_model_comparison_refit.py",
        "c19_recursive_sensitivity.py", "c20_nuc_sensitivity.py",
        "c21_events_in_mean.py", "verify_release.py", "verify_tables.py",
    ]
    require(all((HERE / name).is_file() for name in expected_scripts), "one or more c1-c21 scripts are missing")
    require((ROOT / "requirements.txt").is_file(), "requirements.txt is missing")

    manuscript = (ROOT / "main.tex").read_text()
    for stale in ["p=0.376", "p=0.372", "being annotated", "progressively-more-correct",
                  "sidesteps this entirely", "survives every \\emph{statistical} objection",
                  "by $0.65$ AIC", "& 13349 &", "& 13785 &",
                  "a rank-based estimate", "The lead switches on",
                  "XRP vanishes entirely", "XRP's association is absent",
                  "XRP's lead is absent", "XRP's lead is confined",
                  "The reported \\emph{two-sided} value is the absolute-statistic",
                  "preserving both linear dependence and copula tail dependence"]:
        require(stale not in manuscript, f"stale manuscript phrase remains: {stale}")
    for current in ["$+8.69$", "$p=0.202$", "$3.49\\times$",
                    "scripts \\texttt{c1}--\\texttt{c21}",
                    "union of the six asset calendars",
                    "stronger than the composite mean null",
                    "one-sided dispersion-statistic effective-df tail area",
                    "baseline Student-$t$ bootstrap retains all $2000/2000$ draws",
                    "common calendar (beginning 2 September 2019)",
                    "not labelled a generic two-sided test"]:
        require(current in manuscript, f"current manuscript anchor missing: {current}")

    for name in ["abstract-plaintext.txt", "ssrn-abstract.txt", "arxiv-form-fields.txt"]:
        text = (ROOT / name).read_text()
        for stale in ["4.88", "7.19", "0.283", "p ≈ 0.32", "under review at Digital Finance"]:
            require(stale not in text, f"stale submission text remains in {name}: {stale}")
        require("sharp per-asset-equality null" in text,
                f"{name} does not scope the fitted second-moment null")

    replication_guide = (HERE / "README.md").read_text()
    require("six-asset common calendar" in replication_guide and
            "24 June through" in replication_guide,
            "replication guide omits the daily-sentiment alignment convention")

    dual = (ROOT / "dual-publication-statement.txt").read_text()
    require("10.5281/zenodo.20636796" in dual and "10.5281/zenodo.18099609" not in dual,
            "dual-publication statement cites a stale companion-preprint deposit")

    availability = (ROOT / "data-availability-statement.txt").read_text()
    require("are to be updated" in availability and "earlier public snapshots" in availability,
            "data-availability statement does not disclose the unsynchronised public records")

    require(verify_tables() == 87, "manuscript table verifier did not complete 87 checks")
    require((ROOT / "main.pdf").is_file(), "compiled main.pdf is missing")
    require((ROOT / "main.pdf").stat().st_mtime >= (ROOT / "main.tex").stat().st_mtime,
            "main.pdf predates main.tex; rebuild before release")

    print("PASS: release artifacts are internally consistent, including 87/87 table cells")


if __name__ == "__main__":
    main()
