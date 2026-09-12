"""
C2: Relaxed-Threshold Sensitivity Analysis
==========================================

Re-estimates the GJR-GARCH-X model across four specifications of the
candidate-event pool. Outputs the multiplier delta_infra / delta_reg
and significance under each.

Specifications:
  S1: Baseline. The surviving 50 events in code/data/events.csv.
  S2: Relaxed threshold. C1 candidate pool, ≥1 asset > 1 SD in [-1,+1]
      window. Drops only candidates with no detectable cross-asset response.
  S3: No filter. C1 candidate pool, no Stage-2 impact filter at all.
  S4: Strict. C1 candidate pool, ≥3 assets > 1 SD.

For each spec, the GJR-GARCH-X estimator is fit per asset, the per-asset
infra and reg coefficients are extracted, and the cross-asset means are
recomputed (2.044 vs 0.586 in this pipeline's reconstructed S1 baseline).

Outputs (in the package results directory):
    c2-relaxed-threshold-results.csv
    c2-multiplier-decay.png
    c2-summary.md
"""

import sys
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent  # event-study/
# Data location: prefer a data/ directory beside this file (flat replication
# package layout); otherwise fall back to <root>/code/data (working-tree layout).
CODE_DIR = _CD if (_CD := Path(__file__).resolve().parent).joinpath("data").is_dir() else ROOT / "code"
DATA_DIR = CODE_DIR / "data"
# Output location. Works in both layouts: the working tree (<root>/r1-revision/code/)
# writes to <root>/r1-revision/, while a flat replication package (<pkg>/code/)
# writes to <pkg>/results/. Override with CES_OUT_DIR.
import os as _os
_here = Path(__file__).resolve().parent
OUT_DIR = Path(_os.environ.get("CES_OUT_DIR",
               _here.parent if _here.parent.name == "r1-revision" else _here.parent / "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
from tarch_x_manual import TARCHXEstimator  # noqa: E402

ASSETS = ["btc", "eth", "xrp", "bnb", "ltc", "ada"]
START_DATE = "2019-01-01"
END_DATE = "2025-08-31"
WINDOW_DAYS_BEFORE = 3
WINDOW_DAYS_AFTER = 3
SENTIMENT_CUTOFF = pd.Timestamp("2019-06-01")  # consistent with merge_sentiment_data


# -----------------------------------------------------------------------------
# Data loading (replicate code/data_preparation.py logic, simplified)
# -----------------------------------------------------------------------------
def load_returns_panel():
    """Load winsorized log returns (in %, matching the existing pipeline)."""
    panel = {}
    for a in ASSETS:
        df = pd.read_csv(DATA_DIR / f"{a}.csv")
        df["date"] = pd.to_datetime(df["snapped_at"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        df = df.loc[START_DATE:END_DATE]
        # log returns × 100 (in %)
        logret = np.log(df["price"]).diff() * 100
        # Apply this sensitivity pipeline's global 0.5% / 99.5% clip.
        lo, hi = logret.quantile([0.005, 0.995])
        ret_w = logret.clip(lo, hi)
        panel[a] = ret_w.dropna()
    return panel


def load_sentiment_daily(date_index):
    """Load GDELT sentiment and forward-fill weekly→daily, matching merge_sentiment_data."""
    df = pd.read_csv(DATA_DIR / "gdelt.csv")
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True).dt.tz_convert(None).dt.normalize()
    df = df.set_index("week_start").sort_index()

    # Recompute S_gdelt_normalized with 52-week rolling z-score (min_periods=26)
    win, mp = 52, 26
    rmean = df["S_gdelt_raw"].rolling(win, min_periods=mp).mean()
    rstd = df["S_gdelt_raw"].rolling(win, min_periods=mp).std()
    df["S_gdelt_normalized"] = (df["S_gdelt_raw"] - rmean) / rstd

    # Decomposed sentiments: raw * proportion (matching existing logic if needed)
    if "S_reg_decomposed" not in df.columns or df["S_reg_decomposed"].isna().any():
        df["S_reg_decomposed"] = df["S_gdelt_normalized"] * df["reg_proportion"]
    if "S_infra_decomposed" not in df.columns or df["S_infra_decomposed"].isna().any():
        df["S_infra_decomposed"] = df["S_gdelt_normalized"] * df["infra_proportion"]

    cols = ["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]
    daily = df[cols].reindex(
        pd.date_range(df.index.min(), date_index.max(), freq="D")
    ).ffill()
    daily.loc[daily.index < SENTIMENT_CUTOFF] = 0

    out = pd.DataFrame(index=date_index)
    out = out.join(daily, how="left")
    out = out.fillna(0)
    return out


def build_event_dummies(date_index, candidate_dates_infra, candidate_dates_reg,
                        window_before=3, window_after=3):
    """
    Construct D_infrastructure and D_regulatory aggregate dummies as the max
    over the per-event windows. Matches the existing pipeline's aggregate
    construction (max of per-event dummies).
    """
    n = len(date_index)
    idx = pd.DatetimeIndex(date_index)
    d_inf = pd.Series(0.0, index=idx, name="D_infrastructure")
    d_reg = pd.Series(0.0, index=idx, name="D_regulatory")

    for dt in candidate_dates_infra:
        dt = pd.to_datetime(dt).normalize()
        start = dt - pd.Timedelta(days=window_before)
        end = dt + pd.Timedelta(days=window_after)
        mask = (idx >= start) & (idx <= end)
        d_inf.loc[mask] = 1.0

    for dt in candidate_dates_reg:
        dt = pd.to_datetime(dt).normalize()
        start = dt - pd.Timedelta(days=window_before)
        end = dt + pd.Timedelta(days=window_after)
        mask = (idx >= start) & (idx <= end)
        d_reg.loc[mask] = 1.0

    return pd.DataFrame({"D_infrastructure": d_inf, "D_regulatory": d_reg})


# -----------------------------------------------------------------------------
# Spec definitions
# -----------------------------------------------------------------------------
def get_event_dates_for_spec(spec, baseline_events_df, census_df):
    """
    Return (infra_dates, reg_dates) for a given specification.

    S1 baseline: surviving 50 events
    S2 relaxed:  candidates passing the relaxed threshold (≥1 asset, 1 SD)
    S3 nofilter: ALL candidates (no impact filter)
    S4 strict:   candidates passing the strict threshold (≥3 assets, 1 SD)
    """
    if spec == "S1_baseline":
        inf = baseline_events_df.loc[baseline_events_df["type"] == "Infrastructure", "date"]
        reg = baseline_events_df.loc[baseline_events_df["type"] == "Regulatory", "date"]
        return inf.tolist(), reg.tolist()

    if spec == "S2_relaxed":
        mask = census_df["stage2_relaxed_pass"]
    elif spec == "S3_nofilter":
        mask = census_df["stage2_nofilter_pass"]
    elif spec == "S4_strict":
        mask = census_df["stage2_strict_pass"]
    else:
        raise ValueError(spec)

    sub = census_df.loc[mask]
    inf = sub.loc[sub["tentative_category"] == "Infrastructure", "date"].tolist()
    reg = sub.loc[sub["tentative_category"] == "Regulatory", "date"].tolist()
    return inf, reg


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def main():
    print("Loading returns panel and sentiment...")
    panel = load_returns_panel()
    common_index = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sentiment = load_sentiment_daily(common_index)

    # Baseline (surviving 50)
    events_df = pd.read_csv(DATA_DIR / "events.csv")
    events_df["date"] = pd.to_datetime(events_df["date"])

    # C1 census
    census_df = pd.read_csv(OUT_DIR / "c1-dropout-census.csv")
    census_df["date"] = pd.to_datetime(census_df["date"])

    specs = ["S1_baseline", "S2_relaxed", "S3_nofilter", "S4_strict"]

    rows_per_asset = []
    summary_rows = []

    for spec in specs:
        inf_dates, reg_dates = get_event_dates_for_spec(spec, events_df, census_df)
        print(f"\n=== Spec {spec}: {len(inf_dates)} infra, {len(reg_dates)} reg ===")

        infra_coefs = []
        reg_coefs = []
        infra_se = []
        reg_se = []
        infra_p = []
        reg_p = []
        diagnostics = []

        for a in ASSETS:
            returns = panel[a].copy()
            returns = returns.loc[returns.index >= pd.Timestamp(START_DATE)]
            dummies = build_event_dummies(returns.index, inf_dates, reg_dates,
                                          window_before=WINDOW_DAYS_BEFORE,
                                          window_after=WINDOW_DAYS_AFTER)
            sent = sentiment.reindex(returns.index).fillna(0)
            exog = pd.concat([
                dummies[["D_infrastructure", "D_regulatory"]],
                sent[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]],
            ], axis=1).fillna(0)

            # Fit GJR-GARCH-X
            est = TARCHXEstimator(returns, exog)
            res = est.estimate(method="SLSQP", max_iter=2000)
            if not res.converged:
                print(f"  [WARN] {a} did not converge cleanly under {spec}")
            d_inf = res.params.get("D_infrastructure", np.nan)
            d_reg = res.params.get("D_regulatory", np.nan)
            se_inf = res.std_errors.get("D_infrastructure", np.nan)
            se_reg = res.std_errors.get("D_regulatory", np.nan)
            p_inf = res.pvalues.get("D_infrastructure", np.nan)
            p_reg = res.pvalues.get("D_regulatory", np.nan)

            infra_coefs.append(d_inf)
            reg_coefs.append(d_reg)
            infra_se.append(se_inf)
            reg_se.append(se_reg)
            infra_p.append(p_inf)
            reg_p.append(p_reg)
            diagnostics.append({
                "spec": spec, "asset": a,
                "n_infra_events": len(inf_dates), "n_reg_events": len(reg_dates),
                "converged": res.converged,
                "log_lik": res.log_likelihood, "aic": res.aic,
                "omega": res.params.get("omega", np.nan),
                "alpha": res.params.get("alpha", np.nan),
                "gamma": res.params.get("gamma", np.nan),
                "beta": res.params.get("beta", np.nan),
                "nu": res.params.get("nu", np.nan),
                "D_infrastructure": d_inf,
                "D_regulatory": d_reg,
                "se_infrastructure": se_inf,
                "se_regulatory": se_reg,
                "p_infrastructure": p_inf,
                "p_regulatory": p_reg,
                "S_gdelt": res.params.get("S_gdelt_normalized", np.nan),
                "S_reg": res.params.get("S_reg_decomposed", np.nan),
                "S_infra": res.params.get("S_infra_decomposed", np.nan),
            })

        # Cross-asset summary for this spec
        infra_arr = np.array(infra_coefs, dtype=float)
        reg_arr = np.array(reg_coefs, dtype=float)
        inf_mean = np.nanmean(infra_arr)
        reg_mean = np.nanmean(reg_arr)
        inf_med = np.nanmedian(infra_arr)
        reg_med = np.nanmedian(reg_arr)
        multiplier = inf_mean / reg_mean if reg_mean != 0 else np.nan
        # Welch t-test infra vs reg
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(infra_arr, reg_arr, equal_var=False, nan_policy="omit")
        # Cohen's d
        s_pooled = np.sqrt((np.nanvar(infra_arr, ddof=1) + np.nanvar(reg_arr, ddof=1)) / 2)
        cohens_d = (inf_mean - reg_mean) / s_pooled if s_pooled > 0 else np.nan

        print(f"  Cross-asset mean infra: {inf_mean:.4f}")
        print(f"  Cross-asset mean reg:   {reg_mean:.4f}")
        print(f"  Multiplier:             {multiplier:.3f}x")
        print(f"  Welch t:                {t_stat:.3f} (p={p_val:.4f})")
        print(f"  Cohen's d:              {cohens_d:.3f}")

        summary_rows.append({
            "spec": spec,
            "n_infra_events": len(inf_dates),
            "n_reg_events": len(reg_dates),
            "mean_infra_coef": inf_mean,
            "mean_reg_coef": reg_mean,
            "median_infra_coef": inf_med,
            "median_reg_coef": reg_med,
            "multiplier": multiplier,
            "cohens_d": cohens_d,
            "welch_t": t_stat,
            "welch_p": p_val,
        })
        rows_per_asset.extend(diagnostics)

    # Save results
    df_per_asset = pd.DataFrame(rows_per_asset)
    df_summary = pd.DataFrame(summary_rows)
    df_per_asset.to_csv(OUT_DIR / "c2-relaxed-threshold-results.csv", index=False)
    df_summary.to_csv(OUT_DIR / "c2-summary-table.csv", index=False)
    print(f"\nSaved per-asset: {OUT_DIR/'c2-relaxed-threshold-results.csv'}")
    print(f"Saved summary:   {OUT_DIR/'c2-summary-table.csv'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sp_labels = ["Baseline\n(n=50)", "Relaxed\n(≥1 asset)", "No filter\n(all candidates)", "Strict\n(≥3 assets)"]
    multipliers = df_summary["multiplier"].values
    pvals = df_summary["welch_p"].values

    ax = axes[0]
    bars = ax.bar(sp_labels, multipliers, color=["#7a1f3d", "#a83a5b", "#c25a78", "#d97a93"])
    for i, (b, p) in enumerate(zip(bars, pvals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                f"{multipliers[i]:.2f}x\n({sig})", ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("δ_infra / δ_reg multiplier")
    ax.set_title("Multiplier across event-pool specifications\n"
                 "(descriptive Welch p-values)", fontsize=10)
    ax.set_ylim(0, max(4.25, float(np.nanmax(multipliers)) + 0.7))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    x = np.arange(4)
    width = 0.35
    inf_means = df_summary["mean_infra_coef"].values
    reg_means = df_summary["mean_reg_coef"].values
    ax.bar(x - width / 2, inf_means, width, label="Infrastructure δ̄", color="#7a1f3d")
    ax.bar(x + width / 2, reg_means, width, label="Regulatory δ̄", color="#c8b58c")
    ax.set_xticks(x)
    ax.set_xticklabels(sp_labels)
    ax.set_ylabel("Mean coefficient (cross-asset)")
    ax.set_title("Per-leg means across specifications")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "c2-multiplier-decay.png", dpi=150)
    fig.savefig(OUT_DIR / "c2-multiplier-decay.pdf")
    print(f"Saved plot:      {OUT_DIR/'c2-multiplier-decay.png'}")

    # Markdown summary
    lines = ["# C2 — Relaxed-Threshold Sensitivity: Summary\n"]
    lines.append("**Headline question.** How does the C2 pipeline's mechanically "
                 "reconstructed S1 multiplier change when the Stage-2 "
                 "impact filter is applied mechanically, relaxed, or removed?\n")
    lines.append("**Answer.** See the table below — the multiplier under each specification.\n")
    lines.append("## Cross-Asset Multiplier by Specification\n")
    lines.append("| Spec | N infra | N reg | δ̄ infra | δ̄ reg | Multiplier | Cohen's d | Welch t | p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df_summary.iterrows():
        lines.append(
            f"| {r['spec']} | {int(r['n_infra_events'])} | {int(r['n_reg_events'])} | "
            f"{r['mean_infra_coef']:.4f} | {r['mean_reg_coef']:.4f} | "
            f"**{r['multiplier']:.3f}×** | {r['cohens_d']:.3f} | "
            f"{r['welch_t']:.3f} | {r['welch_p']:.4f} |"
        )

    # Decision-tree interpretation
    relaxed_mult = df_summary.loc[df_summary["spec"] == "S2_relaxed", "multiplier"].iloc[0]
    nofilter_mult = df_summary.loc[df_summary["spec"] == "S3_nofilter", "multiplier"].iloc[0]
    strict_mult = df_summary.loc[df_summary["spec"] == "S4_strict", "multiplier"].iloc[0]
    relaxed_p = df_summary.loc[df_summary["spec"] == "S2_relaxed", "welch_p"].iloc[0]

    lines.append("\n## Interpretation\n")
    lines.append("The mechanical specifications are descriptive inclusion-rule sensitivities. "
                 "Their Welch p-values treat the six per-asset coefficients as independent and "
                 "are not the paper's final inference. The attenuation relative to the curated "
                 "sample is a scope condition; it does not identify selection bias in upstream "
                 "manual curation.\n")
    lines.append(f"- Relaxed-threshold multiplier: **{relaxed_mult:.3f}×** (p = {relaxed_p:.4f})")
    lines.append(f"- No-filter multiplier:         **{nofilter_mult:.3f}×**")
    lines.append(f"- Strict-threshold multiplier:  **{strict_mult:.3f}×**\n")

    lines.append("## Files\n")
    lines.append("- Per-asset model parameters: `c2-relaxed-threshold-results.csv`")
    lines.append("- Cross-asset summary: `c2-summary-table.csv`")
    lines.append("- Plot: `c2-multiplier-decay.png` / `.pdf`\n")

    (OUT_DIR / "c2-summary.md").write_text("\n".join(lines))
    print(f"Saved summary:   {OUT_DIR/'c2-summary.md'}")


if __name__ == "__main__":
    main()
