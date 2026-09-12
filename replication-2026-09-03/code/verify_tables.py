#!/usr/bin/env python3
"""Verify decision-relevant manuscript table cells against committed CSVs."""

from pathlib import Path
import re

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def verify_tables():
    tex = (ROOT / "main.tex").read_text()
    failures = []
    checks = 0

    def indexed_source(frame, key, expected, label):
        if key not in frame.columns:
            failures.append(f"{label}: missing source index column {key}")
            return None
        if frame[key].duplicated().any():
            failures.append(f"{label}: duplicate source rows for {key}")
            return None
        actual = frame[key].astype(str).tolist()
        if actual != list(expected):
            failures.append(f"{label}: source row names/order={actual}")
            return None
        return frame.set_index(key, drop=False)

    def table_block(table_label):
        marker = rf"\label{{{table_label}}}"
        pos = tex.find(marker)
        if pos < 0:
            failures.append(f"{table_label}: table label not found")
            return ""
        start = tex.rfind(r"\begin{table}", 0, pos)
        end = tex.find(r"\end{table}", pos)
        if start < 0 or end < 0:
            failures.append(f"{table_label}: table boundaries not found")
            return ""
        return tex[start:end + len(r"\end{table}")]

    def check(label, expected, pattern, decimals=3, source=None):
        nonlocal checks
        checks += 1
        match = re.search(pattern, tex if source is None else source)
        if not match:
            failures.append(f"{label}: manuscript pattern not found")
            return
        reported = float(match.group(1))
        if f"{reported:.{decimals}f}" != f"{float(expected):.{decimals}f}":
            failures.append(
                f"{label}: manuscript={reported} source={float(expected):.{decimals}f}"
            )

    def check_reported(label, expected, reported, decimals=3):
        nonlocal checks
        checks += 1
        if reported is None:
            failures.append(f"{label}: manuscript value not found")
            return
        if f"{float(reported):.{decimals}f}" != f"{float(expected):.{decimals}f}":
            failures.append(
                f"{label}: manuscript={float(reported)} source={float(expected):.{decimals}f}"
            )

    def row_cells(source, row_pattern, label):
        match = re.search(rf"(?m)^{row_pattern}[^\n]*\\\\\s*$", source)
        if not match:
            failures.append(f"{label}: manuscript row not found")
            return []
        return [cell.strip() for cell in match.group(0).rsplit(r"\\", 1)[0].split("&")]

    def numbers(cell):
        return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", cell)]

    # Per-asset GARCH estimates and diagnostics (Tables 4 and 9).
    c14 = pd.read_csv(RESULTS / "c14-garch-diagnostics-per-asset.csv")
    expected_assets = ["btc", "eth", "xrp", "bnb", "ltc", "ada"]
    if c14["asset"].astype(str).tolist() != expected_assets or c14["asset"].duplicated().any():
        failures.append("c14: asset source rows/order drifted")
    for _, row in c14.iterrows():
        asset = row.asset.upper()
        check(
            f"T4 {asset} delta_infra",
            row.delta_infra,
            rf"{asset} & [\d.]+ & [\d.]+ & \$-?[\d.]+\$ & [\d.]+ & "
            rf"[\d.]+ & [\d.]+ & ([\d.]+) & [\d.]+ \\",
        )
        check(
            f"T4 {asset} delta_reg",
            row.delta_reg,
            rf"{asset} & [\d.]+ & [\d.]+ & \$-?[\d.]+\$ & [\d.]+ & "
            rf"[\d.]+ & [\d.]+ & [\d.]+ & ([\d.]+) \\",
        )
        check(
            f"T4 {asset} omega",
            row.omega,
            rf"{asset} & ([\d.]+) & [\d.]+ & \$-?[\d.]+\$ & [\d.]+ & "
            rf"[\d.]+ & [\d.]+ & [\d.]+ & [\d.]+ \\",
        )
        check(
            f"T9 {asset} LB p5",
            row.LB_z2_p5_adj,
            rf"{asset} & ([\d.]+) & [\d.]+ & [\d.]+ & [\d.]+ & [\d.]+ \\",
        )
        check(
            f"T9 {asset} ARCH-LM p5",
            row.ARCHLM_p5,
            rf"{asset} & [\d.]+ & [\d.]+ & [\d.]+ & ([\d.]+) & [\d.]+ \\",
        )

    check(
        "T4 mean delta_infra",
        c14.delta_infra.mean(),
        r"Mean & --- & --- & --- & --- & --- & --- & \\textbf\{([\d.]+)\} "
        r"& \\textbf\{[\d.]+\} \\",
    )
    check(
        "T4 mean delta_reg",
        c14.delta_reg.mean(),
        r"Mean & --- & --- & --- & --- & --- & --- & \\textbf\{[\d.]+\} "
        r"& \\textbf\{([\d.]+)\} \\",
    )

    # First-moment headline table.
    c11_frame = pd.read_csv(RESULTS / "c-gate-returns-unified-results.csv")
    c11_bases = ["A: 6-asset (incl XRP)", "B: 5-asset (ex XRP)"]
    c11 = indexed_source(c11_frame, "basis", c11_bases, "c11")
    if c11 is None:
        raise AssertionError("\n".join(failures))
    six = c11.loc[c11_bases[0]]
    five = c11.loc[c11_bases[1]]
    check(
        "T2 six-asset difference",
        six.diff_pp,
        r"Six-asset \(incl\.\\ XRP\), \$n=26/24\$ & \$-[\d.]+\\%\$ & "
        r"\$-[\d.]+\\%\$ & \$\+([\d.]+)\$",
        2,
    )
    check(
        "T2 six-asset block p",
        six.block_p_two,
        r"Six-asset \(incl\.\\ XRP\).*?& \$([\d.]+)\$ & \$[\d.]+\$ \\",
    )
    check(
        "T2 six-asset Welch p",
        six.im_p,
        r"Six-asset \(incl\.\\ XRP\).*?& \$[\d.]+\$ & \$([\d.]+)\$ \\",
    )
    check(
        "T2 five-asset difference",
        five.diff_pp,
        r"Five-asset \(ex-XRP\), \$n=26/24\$ & \$\+[\d.]+\\%\$ & "
        r"\$-[\d.]+\\%\$ & \$\+([\d.]+)\$",
        2,
    )

    # Fixed-path t-copula comparison tables.
    c9_frame = pd.read_csv(RESULTS / "c9-tcopula-results.csv")
    c9 = indexed_source(c9_frame, "spec", ["baseline", "crisis", "full"], "c9")
    if c9 is None:
        raise AssertionError("\n".join(failures))
    tcopula_table = table_block("tab:tcopula")
    compare_table = table_block("tab:tcopula_compare")
    compare_rows = {
        "baseline": r"Baseline \(no break controls\)",
        "crisis": r"\$\+\$ High-variance-regime control",
        "full": r"\$\+\$ Full regime controls",
    }
    for spec, row_pattern in compare_rows.items():
        cells = row_cells(tcopula_table, row_pattern, f"T5 {spec}")
        if len(cells) != 5:
            failures.append(f"T5 {spec}: expected 5 cells, found {len(cells)}")
            reported = [None] * 4
        else:
            reported = [numbers(cell)[0] for cell in cells[1:]]
        row = c9.loc[spec]
        mean_reg = row.d_bar_obs / (row.multiplier - 1.0)
        mean_infra = row.multiplier * mean_reg
        expected = [mean_infra, mean_reg, row.multiplier, row.p_tcopula_two_sided]
        for field_label, source_value, manuscript_value, decimals in zip(
            ("mean infra", "mean reg", "multiplier", "absolute p"),
            expected,
            reported,
            (3, 3, 2, 3),
        ):
            check_reported(
                f"T5 {spec} {field_label}", source_value, manuscript_value, decimals
            )
    for spec, row_pattern in compare_rows.items():
        cells = row_cells(compare_table, row_pattern, f"T7 {spec}")
        if len(cells) != 6:
            failures.append(f"T7 {spec}: expected 6 cells, found {len(cells)}")
            values = [None] * 7
        else:
            values = [
                numbers(cells[1])[0], numbers(cells[2])[0], numbers(cells[3])[0],
                *numbers(cells[4]), numbers(cells[5])[0],
            ]
        row = c9.loc[spec]
        expected = [
            row.multiplier, row.p_gaussian_old_one_sided,
            row.p_tcopula_one_sided, row.null_sd_gaussian,
            row.null_sd_tcopula, row.rho_resid,
        ]
        labels = ("multiplier", "Gaussian p", "Student-t p", "Gaussian SD",
                  "Student-t SD", "rho_resid")
        for field_label, source_value, manuscript_value in zip(labels, expected, values):
            decimals = 2 if field_label == "multiplier" else 3
            check_reported(
                f"T7 {spec} {field_label}", source_value, manuscript_value, decimals
            )

    # Fitted-DGP internal size table: both rejection rates and MC standard errors.
    c10_frame = pd.read_csv(RESULTS / "c10-size-study-results.csv")
    c10_methods = ["naive_iid", "design_effect", "gaussian_copula_boot", "tcopula_boot"]
    c10 = indexed_source(c10_frame, "method", c10_methods, "c10")
    if c10 is None:
        raise AssertionError("\n".join(failures))
    size_table = table_block("tab:size_study")
    size_rows = {
        "naive_iid": r"\(i\) Naive i\.i\.d\.\\ \$t\$-test",
        "design_effect": r"\(ii\) Design-effect, \$t\(N\{-\}1\{=\}5\)\$ critical value",
        "gaussian_copula_boot": r"\(iii\) Gaussian-copula bootstrap",
        "tcopula_boot": r"\(iv\) \\textbf\{Student-\$t\$-copula bootstrap\}",
    }
    for method, row_pattern in size_rows.items():
        cells = row_cells(size_table, row_pattern, f"T8 {method}")
        if len(cells) != 3 or len(numbers(cells[1])) < 2 or len(numbers(cells[2])) < 2:
            failures.append(f"T8 {method}: malformed size/SE cells")
            reported = [None] * 4
        else:
            reported = [*numbers(cells[1])[:2], *numbers(cells[2])[:2]]
        row = c10.loc[method]
        expected = [row.size_005, row.se_005, row.size_010, row.se_010]
        for suffix, source_value, manuscript_value in zip(
            ("size05", "se05", "size10", "se10"), expected, reported
        ):
            check_reported(f"T8 {method} {suffix}", source_value, manuscript_value)

    # DGP perturbation table.
    c10b_frame = pd.read_csv(RESULTS / "c10b-dgp-perturbation-results.csv")
    c10b_scenarios = [
        "S0 baseline (fitted nu, matched)",
        "S1 lighter tails nu=8 (matched)",
        "S2 heavier tails nu=2.5 (matched)",
        "S3 near-Gaussian t-copula (nu_c=200, fitted margins)",
        "S4 MIS-SPECIFIED (simulate nu=2.5, calibrate at fitted nu)",
    ]
    c10b = indexed_source(c10b_frame, "scenario", c10b_scenarios, "c10b")
    if c10b is None:
        raise AssertionError("\n".join(failures))
    check(
        "T11 fitted-nu naive",
        c10b.loc[c10b_scenarios[0]].naive_iid_size05,
        r"Fitted \$\\nu\$ \(baseline\) "
        r"& ([\d.]+) &",
        2,
    )
    check(
        "T11 fitted-nu t-copula",
        c10b.loc[c10b_scenarios[0]].tcopula_boot_size05,
        r"Fitted \$\\nu\$ .*? & [\d.]+ & [\d.]+ & [\d.]+ & "
        r"\\textbf\{([\d.]+)\} \\",
        2,
    )
    check(
        "T11 misspecified t-copula",
        c10b.loc[c10b_scenarios[4]].tcopula_boot_size05,
        r"Tail mismatch \(DGP \$\\nu=2\.5\$\) & "
        r"[\d.]+ & [\d.]+ & [\d.]+ & \\textit\{([\d.]+)\} \\",
        2,
    )

    # Event-screen sensitivity table.
    c2_frame = pd.read_csv(RESULTS / "c2-summary-table.csv")
    c2 = indexed_source(
        c2_frame,
        "spec",
        ["S1_baseline", "S2_relaxed", "S3_nofilter", "S4_strict"],
        "c2",
    )
    if c2 is None:
        raise AssertionError("\n".join(failures))
    check(
        "T12 curated multiplier",
        c2.loc["S1_baseline"].multiplier,
        r"Primary curated \(\$n=50\$\)  & 26 & 24 & 2\.044 & 0\.586 & "
        r"\\textbf\{([\d.]+)\$\\times\$\}",
        2,
    )
    check(
        "T12 no-filter multiplier",
        c2.loc["S3_nofilter"].multiplier,
        r"No filter \(135\)           & 82 & 53 & [\d.]+ & [\d.]+ & "
        r"([\d.]+)\$\\times\$",
        2,
    )

    if checks != 87:
        failures.append(f"internal verifier error: expected 87 checks, ran {checks}")
    if failures:
        raise AssertionError("\n".join(failures))
    return checks


def main():
    checks = verify_tables()
    print(f"PASS: {checks}/{checks} manuscript table values match committed CSVs")


if __name__ == "__main__":
    main()
