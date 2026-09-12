"""
C8h: intermediate Gaussian-innovation break-control bootstrap.
================================================================

This script applies c7's fixed-path Gaussian-innovation bootstrap to the full
regime and the asset-specific high-variance-segment control (legacy machine key
``crisis``). It is retained as a diagnostic comparator,
not as the paper's final inference: c9 replaces the Gaussian draw with fitted
Student-t margins and adds the full seed-indexed audit schema. Both scripts
inherit c7's deterministic multistart rescue. The current c8h artifacts are
retained only as intermediate Gaussian comparators to the corresponding c9
rows.

Method (identical to c7's fixed-path parametric null-imposed test):
  * Per asset, the variance-equation exog is [D_infra, D_reg, S_gdelt, S_reg,
    S_infra, <regime dummies>]. Keeping D_infra, D_reg as the FIRST two exog
    columns means params[5], params[6] remain the infra/reg deltas, so c7's
    worker functions (which read p[5]-p[6]) apply unchanged.
  * NULL fit imposes delta_infra = delta_reg via a single combined event dummy
    D_event = D_infra + D_reg, WHILE KEEPING the regime dummies:
        exog_null = [D_event, S_gdelt, S_reg, S_infra, <regime dummies>].
  * Simulate z* ~ MVN(0, R_z) on the standardised-residual correlation, build
    returns* from the null variance path, refit UNRESTRICTED (regime dummies
    still in), collect d_bar_b = mean_a(delta_infra - delta_reg). One-sided p =
    P(d_bar_b >= d_bar_obs).

Reuses c7's _GLOBAL, worker functions, and run_bootstrap by repopulating c7's
module globals with the regime-augmented design -- so the bootstrap engine is
byte-for-byte the c7 engine, only the design matrices change.

Outputs are written to the package results directory (or `CES_OUT_DIR`).
"""
import argparse
import hashlib
import sys
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_relaxed_threshold_sensitivity as c2          # loaders
import c7_ccc_garchx_bootstrap as c7                    # bootstrap engine + workers
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA        # estimator

ASSETS = c2.ASSETS
SENT_COLS = ["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]
N_STARTS_FIT = 6
MAX_ITER = 2000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Regime-dummy construction (same logic as c8a, factored out here)
# ---------------------------------------------------------------------------
def regime_dummies(date_index, asset, c3_df, variant):
    """Return an (n x k) np.ndarray of regime dummies for `asset`.

    variant="full":   one dummy per cond_variance regime segment after the first
                      (first segment = baseline, dropped).
    variant="crisis": legacy machine key for one asset-specific high-variance
                      segment (last 2021 break -> first >=2022 break). Four
                      windows end around FTX; BNB and ADA end in July 2022.
    """
    idx = pd.DatetimeIndex(date_index)
    breaks = sorted(pd.Timestamp(b) for b in
                    c3_df[(c3_df["asset"] == asset) &
                          (c3_df["series_type"] == "cond_variance")]["break_date"])
    if variant == "full":
        edges = [idx.min()] + breaks + [idx.max() + pd.Timedelta(days=1)]
        cols = []
        for i in range(1, len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            d = np.zeros(len(idx))
            d[(idx >= lo) & (idx < hi)] = 1.0
            cols.append(d)
        return np.column_stack(cols) if cols else np.zeros((len(idx), 0)), None
    elif variant == "crisis":
        starts_2021 = [b for b in breaks if b.year == 2021]
        ends_2022p = [b for b in breaks if b.year >= 2022]
        if starts_2021 and ends_2022p:
            lo, hi = max(starts_2021), min(ends_2022p)
        elif ends_2022p:
            hi = min(ends_2022p)
            prior = [b for b in breaks if b < hi]
            lo = max(prior) if prior else idx.min()
        else:
            lo, hi = idx.min(), idx.max() + pd.Timedelta(days=1)
        d = np.zeros(len(idx))
        d[(idx >= lo) & (idx < hi)] = 1.0
        return d[:, None], (lo, hi)
    raise ValueError(variant)


# ---------------------------------------------------------------------------
# Design with regime dummies appended (D_infra, D_reg kept first)
# ---------------------------------------------------------------------------
def build_design_with_regimes(variant):
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(c2.OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    c3 = pd.read_csv(c2.OUT_DIR / "c3-bai-perron-results.csv"); c3["break_date"] = pd.to_datetime(c3["break_date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)

    design, return_series, crisis_windows = {}, {}, {}
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
        dum = c2.build_event_dummies(r.index, inf_d, reg_d,
                                     c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s = sent.reindex(r.index).fillna(0)
        Dinf = dum["D_infrastructure"].values
        Dreg = dum["D_regulatory"].values
        S = s[SENT_COLS].values
        reg_d_mat, win = regime_dummies(r.index, a, c3, variant)
        crisis_windows[a] = win
        # CRITICAL: D_infra, D_reg first -> params[5], params[6] stay the deltas
        exog_unr = np.column_stack([Dinf, Dreg, S, reg_d_mat])
        exog_null = np.column_stack([Dinf + Dreg, S, reg_d_mat])
        design[a] = {"returns": r.values.astype(float),
                     "exog_unr": exog_unr, "exog_null": exog_null,
                     "index": r.index, "n_regime": reg_d_mat.shape[1]}
        return_series[a] = r
    ret_df = pd.DataFrame(return_series).dropna()
    return design, inf_d, reg_d, ret_df, crisis_windows


def run_variant(variant, B, n_jobs, seed):
    variant_label = display_variant(variant)
    print(f"\n{'='*70}\n=== VARIANT: {variant_label} ===\n{'='*70}")
    if variant_label != variant:
        print(f"Machine key: {variant}")
    design, inf_d, reg_d, ret_df, crisis_windows = build_design_with_regimes(variant)
    for a in ASSETS:
        print(f"  {a}: n_obs={design[a]['returns'].shape[0]}  n_regime_dummies={design[a]['n_regime']}"
              + (f"  high_variance={crisis_windows[a][0].date()}..{crisis_windows[a][1].date()}"
                 if crisis_windows[a] else ""))

    # observed fits (reuse c7.fit_observed, which reads exog_unr/exog_null and
    # uses p[5]/p[6] as the deltas -- valid because D_infra,D_reg are first)
    print("Fitting observed (unrestricted + null), multistart...")
    t0 = time.time()
    observed = c7.fit_observed(design, seed=seed)
    print(f"  done {time.time()-t0:.1f}s")

    di = np.array([observed[a]["delta_infra"] for a in ASSETS])
    dr = np.array([observed[a]["delta_reg"] for a in ASSETS])
    d_obs = di - dr
    d_bar_obs = float(d_obs.mean())
    multiplier = float(di.mean() / dr.mean())
    print("Per-asset deltas (unrestricted, WITH regime controls):")
    for a, x, y in zip(ASSETS, di, dr):
        print(f"  {a}: dInfra={x:.4f} dReg={y:.4f} diff={x-y:.4f}")
    print(f"  d_bar_obs={d_bar_obs:.4f}  multiplier={multiplier:.3f}x")

    # naive Welch on the 6 per-asset deltas (what c8a reported)
    t_naive, p_naive = stats.ttest_ind(di, dr, equal_var=False)

    # correlations
    R_return = ret_df.corr().values
    rho_return = c7.mean_off_diag(R_return)
    z_df = pd.DataFrame({a: pd.Series(observed[a]["z_resid"], index=design[a]["index"])
                         for a in ASSETS}).dropna()
    R_z = z_df.corr().values
    rho_resid = c7.mean_off_diag(R_z)
    print(f"  rho_return={rho_return:.4f}  rho_resid={rho_resid:.4f}")

    # PD-safe cholesky
    R_z_pd = R_z.copy(); eps_jit = 0.0
    while True:
        try:
            L_z = np.linalg.cholesky(R_z_pd); break
        except np.linalg.LinAlgError:
            eps_jit = max(eps_jit * 10, 1e-8)
            R_z_pd = R_z + eps_jit * np.eye(len(ASSETS))

    common_idx = z_df.index
    common_pos = {a: pd.Index(design[a]["index"]).get_indexer(common_idx) for a in ASSETS}

    # populate c7's module globals so its workers operate on THIS design
    c7._GLOBAL["design"] = design
    c7._GLOBAL["observed"] = observed
    c7._GLOBAL["R_z"] = R_z
    c7._GLOBAL["L_z"] = L_z
    c7._GLOBAL["n_common"] = len(common_idx)
    c7._GLOBAL["common_pos"] = common_pos
    c7._GLOBAL["max_len"] = max(observed[a]["resid_unr"].shape[0] for a in ASSETS)
    c7.install_simulation_calendar(design)

    print(f"[null-imposed CCC bootstrap] B={B} n_jobs={n_jobs}...")
    t0 = time.time()
    null_arr, null_drop = c7.run_bootstrap(c7._draw_parametric_null, B, n_jobs, seed + 10_000)
    Bn = len(null_arr)
    p_one = (np.sum(null_arr >= d_bar_obs) + 1) / (Bn + 1)
    p_two = (np.sum(np.abs(null_arr) >= abs(d_bar_obs)) + 1) / (Bn + 1)
    print(f"  done {time.time()-t0:.1f}s used={Bn} dropped={null_drop} ({null_drop/B:.1%})")
    print(
        f"  ROBUST one-sided p={p_one:.4f}  "
        f"absolute-statistic p={p_two:.4f}"
    )
    print(f"  null d_bar mean={null_arr.mean():.4f} sd={null_arr.std():.4f}")
    print(f"  (naive Welch p={p_naive:.4f} for contrast)")

    np.savez(c2.OUT_DIR / f"c8h-break-ccc-draws-{variant}.npz",
             null=null_arr, d_bar_obs=d_bar_obs,
             simulation_calendar_ns=pd.DatetimeIndex(
                 c7._GLOBAL["simulation_calendar"]
             ).asi8.astype(np.int64),
             availability_pattern_codes=c7._GLOBAL["availability_pattern_codes"],
             availability_pattern_counts=c7._GLOBAL["availability_pattern_counts"])

    return {
        "variant": variant,
        "multiplier": multiplier,
        "d_bar_obs": d_bar_obs,
        "mean_infra": float(di.mean()), "mean_reg": float(dr.mean()),
        "rho_return": rho_return, "rho_resid": rho_resid,
        "n_calendar": int(c7._GLOBAL["n_calendar"]),
        "availability_pattern_codes": ";".join(
            str(int(v)) for v in c7._GLOBAL["availability_pattern_codes"]
        ),
        "availability_pattern_counts": ";".join(
            str(int(v)) for v in c7._GLOBAL["availability_pattern_counts"]
        ),
        "naive_welch_t": float(t_naive), "naive_welch_p": float(p_naive),
        "robust_p_one_sided": float(p_one),
        # Legacy schema name: this is the unrecentred absolute-statistic tail.
        "robust_p_two_sided": float(p_two),
        "null_dbar_mean": float(null_arr.mean()), "null_dbar_sd": float(null_arr.std()),
        "B_used": Bn, "frac_dropped": null_drop / B,
        "per_asset_diff": d_obs.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=22)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    t_start = time.time()
    print(f"numba available: {_HAVE_NUMBA}")
    results = []
    for variant in ["full", "crisis"]:
        results.append(run_variant(variant, args.B, args.n_jobs, args.seed))

    df = pd.DataFrame(results)
    df.to_csv(
        c2.OUT_DIR / "c8h-break-ccc-bootstrap-results.csv",
        index=False,
        float_format="%.17g",
    )
    print(f"\nSaved c8h-break-ccc-bootstrap-results.csv")
    write_finding(results, args.B, time.time() - t_start)
    print(f"TOTAL {(time.time()-t_start)/60:.1f} min")


def verdict(p):
    if p < 0.05:
        return f"Below 5% in this intermediate Gaussian-innovation bootstrap (p={p:.4f})."
    if p < 0.10:
        return f"Between 5% and 10% in this intermediate Gaussian-innovation bootstrap (p={p:.4f})."
    return f"Not below 10% in this intermediate Gaussian-innovation bootstrap (p={p:.4f})."


def display_variant(variant):
    return "asset-specific high-variance" if variant == "crisis" else variant


def write_finding(results, B, elapsed):
    lines = ["# C8h — Intermediate Gaussian-Innovation Break-Control Bootstrap\n"]
    lines.append(f"_Null-imposed CCC-GARCH-X parametric bootstrap, B={B}; runtime {elapsed/60:.1f} min; "
                 f"same engine as c7 (regime dummies added to the variance-equation exog)._\n")
    lines.append("This is a retained Gaussian-innovation comparator. The final matched-margin "
                 "inference is c9; a c8h run with more than 10% discarded draws is reliability-flagged.\n")
    lines.append("## Result\n")
    lines.append("| variant | multiplier | naive Welch p | **robust CCC p (1-sided)** | robust absolute-statistic p | rho_resid | dropped |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        label = display_variant(r["variant"])
        lines.append(f"| {label} regime | {r['multiplier']:.2f}x | {r['naive_welch_p']:.4f} | "
                     f"**{r['robust_p_one_sided']:.4f}** | {r['robust_p_two_sided']:.4f} | "
                     f"{r['rho_resid']:.3f} | {r['frac_dropped']:.1%} |")
    lines.append("")
    for r in results:
        label = display_variant(r["variant"])
        lines.append(f"- **{label} regime dummies** (multiplier {r['multiplier']:.2f}x, "
                     f"d_bar_obs={r['d_bar_obs']:.3f}): {verdict(r['robust_p_one_sided'])} "
                     f"Naive Welch was {r['naive_welch_p']:.4f}; the cross-asset-robust value is "
                     f"{r['robust_p_one_sided']:.4f}.")
    lines.append("")
    lines.append("## Honest verdict\n")
    pf = next(r for r in results if r["variant"] == "full")["robust_p_one_sided"]
    pc = next(r for r in results if r["variant"] == "crisis")["robust_p_one_sided"]
    lines.append(
        f"Gaussian-innovation one-sided p-values are full={pf:.4f} and "
        f"asset-specific high-variance={pc:.4f} (legacy machine key `crisis`). "
        "They are intermediate diagnostics and must not replace c9's Student-t results."
    )
    lines.append("\n## Files\n")
    lines.append("- `c8h-break-ccc-bootstrap-results.csv`, `c8h-break-ccc-draws-{full,crisis}.npz`")
    lines.append("- `code/c8h_break_controls_ccc_bootstrap.py` (reuses `c7_ccc_garchx_bootstrap.py` engine)")
    lines.append("\n## Release provenance\n")
    lines.append(f"- C8h source SHA-256: `{sha256_file(Path(__file__))}`")
    lines.append(
        "- C8h results CSV SHA-256: "
        f"`{sha256_file(c2.OUT_DIR / 'c8h-break-ccc-bootstrap-results.csv')}`"
    )
    for variant in ("full", "crisis"):
        c8h_path = c2.OUT_DIR / f"c8h-break-ccc-draws-{variant}.npz"
        c9_path = c2.OUT_DIR / f"c9-tcopula-draws-{variant}.npz"
        lines.append(
            f"- {display_variant(variant).capitalize()} C8h/C9 draw-archive SHA-256: "
            f"`{sha256_file(c8h_path)}` / `{sha256_file(c9_path)}`"
        )
    lines.append(
        "The release verifier recomputes all C8h tail counts and summaries and "
        "requires its retained vectors, observed statistics, calendar, and availability "
        "patterns to match C9's corresponding Gaussian legs exactly."
    )
    lines.append(
        "The legacy CSV field `robust_p_two_sided` stores the unrecentred "
        "absolute-statistic tail; it is not a generic two-sided p-value."
    )
    (c2.OUT_DIR / "c8h-break-controls-ccc-FINDING.md").write_text("\n".join(lines))
    print(f"Saved c8h-break-controls-ccc-FINDING.md")


if __name__ == "__main__":
    main()
