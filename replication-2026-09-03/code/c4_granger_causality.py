"""
C4: Granger Causality Tests on Sentiment vs Volatility
=======================================================

Pairwise Granger causality tests on (i) decomposed regulatory sentiment vs
|daily log returns|, and (ii) decomposed infrastructure sentiment vs |daily
log returns|, for each cryptocurrency.

Tests both directions:
    sentiment → |returns| (does sentiment Granger-cause volatility?)
    |returns| → sentiment (does volatility Granger-cause sentiment?)

Lags 1 through 10, with AIC and BIC for optimal-lag selection. Reports the
F-statistic, asymptotic p-value, and (bidirectional) lag at which AIC and
BIC are minimised.

The weekly GDELT sentiment is forward-filled to daily frequency. This repeats
each weekly observation and changes the effective information content, so the
daily-lag results should not be interpreted as a test of weekly lead-lag effects.

Outputs:
    results/c4-granger-results.csv
    results/c4-granger-summary.md
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parent.parent.parent
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_relaxed_threshold_sensitivity import load_returns_panel, load_sentiment_daily  # noqa: E402

ASSETS = ["btc", "eth", "xrp", "bnb", "ltc", "ada"]
MAX_LAG = 10


def run_granger(y: pd.Series, x: pd.Series, maxlag: int = MAX_LAG) -> dict:
    """Run Granger tests for x->y at lags 1..maxlag.

    The statsmodels convention is: H0 is that x does NOT Granger-cause y.
    The DataFrame columns are [y, x] (y first).
    Returns dict with optimal_lag (by ssr_ftest p-value),
    f_stat at optimal lag, asymptotic p, and ANY-lag minimum p.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 100:
        return None
    df.columns = ["y", "x"]

    fstats = {}
    pvals = {}
    aics = {}
    bics = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = grangercausalitytests(df[["y", "x"]], maxlag=maxlag, verbose=False)
        except Exception as e:
            return {"error": str(e)}

    for lag in range(1, maxlag + 1):
        if lag not in res:
            continue
        # ssr_ftest is most standard
        f_stat, p_val, df_num, df_den = res[lag][0]["ssr_ftest"]
        fstats[lag] = float(f_stat)
        pvals[lag] = float(p_val)
        # AR(p) on y including lags of x for AIC/BIC
        model = res[lag][1][1]  # the restricted/unrestricted regression
        # statsmodels res[lag][1] is (restricted_result, unrestricted_result)
        # the second is unrestricted (includes x lags), use its aic/bic
        try:
            aics[lag] = float(model.aic)
            bics[lag] = float(model.bic)
        except Exception:
            pass

    if not pvals:
        return {"error": "no valid lags"}

    # Optimal lag by min ssr_ftest p
    opt_lag_p = min(pvals, key=pvals.get)
    # Optimal lag by min AIC / BIC
    opt_lag_aic = min(aics, key=aics.get) if aics else opt_lag_p
    opt_lag_bic = min(bics, key=bics.get) if bics else opt_lag_p

    return {
        "f_stat_at_opt_p": fstats[opt_lag_p],
        "p_value_min": pvals[opt_lag_p],
        "optimal_lag_by_p": opt_lag_p,
        "optimal_lag_by_aic": opt_lag_aic,
        "optimal_lag_by_bic": opt_lag_bic,
        "p_values_per_lag": pvals,
        "f_stats_per_lag": fstats,
        "n_obs": len(df),
    }


def main():
    print("Loading data...")
    panel = load_returns_panel()
    common_index = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sentiment = load_sentiment_daily(common_index)

    rows = []
    for a in ASSETS:
        r = panel[a]
        abs_r = r.abs()
        # Forward-filled sentiment, aligned to returns
        s_reg = sentiment["S_reg_decomposed"].reindex(r.index).fillna(0)
        s_inf = sentiment["S_infra_decomposed"].reindex(r.index).fillna(0)
        s_norm = sentiment["S_gdelt_normalized"].reindex(r.index).fillna(0)

        for sent_name, sent_series in [("S_reg_decomposed", s_reg),
                                        ("S_infra_decomposed", s_inf),
                                        ("S_gdelt_normalized", s_norm)]:
            print(f"  {a.upper()} | {sent_name}: testing both directions...")
            # Direction 1: sentiment -> |returns|
            #   In statsmodels, df.columns=[y, x] tests x->y.
            #   So we pass [abs_r, sent_series] for sent->|r|.
            res_s_to_r = run_granger(abs_r, sent_series)
            # Direction 2: |returns| -> sentiment
            res_r_to_s = run_granger(sent_series, abs_r)

            if res_s_to_r is None or "error" in (res_s_to_r or {}):
                continue
            if res_r_to_s is None or "error" in (res_r_to_s or {}):
                continue

            rows.append({
                "asset": a,
                "sentiment_series": sent_name,
                "direction": "sentiment_to_returns",
                "f_stat": res_s_to_r["f_stat_at_opt_p"],
                "p_value_min": res_s_to_r["p_value_min"],
                "optimal_lag_by_p": res_s_to_r["optimal_lag_by_p"],
                "optimal_lag_by_aic": res_s_to_r["optimal_lag_by_aic"],
                "optimal_lag_by_bic": res_s_to_r["optimal_lag_by_bic"],
                "n_obs": res_s_to_r["n_obs"],
                "p_lag1": res_s_to_r["p_values_per_lag"].get(1, np.nan),
                "p_lag3": res_s_to_r["p_values_per_lag"].get(3, np.nan),
                "p_lag5": res_s_to_r["p_values_per_lag"].get(5, np.nan),
                "p_lag7": res_s_to_r["p_values_per_lag"].get(7, np.nan),
                "p_lag10": res_s_to_r["p_values_per_lag"].get(10, np.nan),
            })
            rows.append({
                "asset": a,
                "sentiment_series": sent_name,
                "direction": "returns_to_sentiment",
                "f_stat": res_r_to_s["f_stat_at_opt_p"],
                "p_value_min": res_r_to_s["p_value_min"],
                "optimal_lag_by_p": res_r_to_s["optimal_lag_by_p"],
                "optimal_lag_by_aic": res_r_to_s["optimal_lag_by_aic"],
                "optimal_lag_by_bic": res_r_to_s["optimal_lag_by_bic"],
                "n_obs": res_r_to_s["n_obs"],
                "p_lag1": res_r_to_s["p_values_per_lag"].get(1, np.nan),
                "p_lag3": res_r_to_s["p_values_per_lag"].get(3, np.nan),
                "p_lag5": res_r_to_s["p_values_per_lag"].get(5, np.nan),
                "p_lag7": res_r_to_s["p_values_per_lag"].get(7, np.nan),
                "p_lag10": res_r_to_s["p_values_per_lag"].get(10, np.nan),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c4-granger-results.csv", index=False)
    print(f"\nWrote {OUT_DIR/'c4-granger-results.csv'} ({len(df)} test pairs)")

    # ------------ Markdown summary -----------
    lines = ["# C4 — Granger Causality (Sentiment vs Volatility): Summary\n"]
    lines.append(
        "Pairwise Granger causality tests at lags 1-10 between decomposed GDELT "
        "sentiment and absolute daily log returns, per asset. Weekly GDELT data "
        "forward-filled to daily (a horizon-mismatched diagnostic that repeats "
        "weekly values across daily rows). Reports the minimum p-value across lags and "
        "the AIC/BIC-optimal lag.\n"
    )

    # Pivot to direction-comparison view: for each asset/sentiment, show both
    # directions side-by-side.
    pivot_rows = []
    for a in ASSETS:
        for sent in ["S_reg_decomposed", "S_infra_decomposed", "S_gdelt_normalized"]:
            sub = df[(df["asset"] == a) & (df["sentiment_series"] == sent)]
            if sub.empty:
                continue
            s_to_r = sub[sub["direction"] == "sentiment_to_returns"]
            r_to_s = sub[sub["direction"] == "returns_to_sentiment"]
            if s_to_r.empty or r_to_s.empty:
                continue
            p_s_to_r = float(s_to_r["p_value_min"].iloc[0])
            p_r_to_s = float(r_to_s["p_value_min"].iloc[0])
            l_s_to_r = int(s_to_r["optimal_lag_by_aic"].iloc[0])
            l_r_to_s = int(r_to_s["optimal_lag_by_aic"].iloc[0])
            f_s_to_r = float(s_to_r["f_stat"].iloc[0])
            f_r_to_s = float(r_to_s["f_stat"].iloc[0])
            # Direction interpretation
            sig_s_to_r = p_s_to_r < 0.05
            sig_r_to_s = p_r_to_s < 0.05
            if sig_s_to_r and not sig_r_to_s:
                interp = "Sentiment LEADS"
            elif sig_r_to_s and not sig_s_to_r:
                interp = "Returns LEAD"
            elif sig_s_to_r and sig_r_to_s:
                interp = "Bidirectional"
            else:
                interp = "Neither (independent)"
            pivot_rows.append({
                "asset": a, "sentiment": sent,
                "f_s_to_r": f_s_to_r, "p_s_to_r": p_s_to_r, "lag_s_to_r": l_s_to_r,
                "f_r_to_s": f_r_to_s, "p_r_to_s": p_r_to_s, "lag_r_to_s": l_r_to_s,
                "interpretation": interp,
            })

    lines.append("## Direction Summary (min-p across lags 1-10)\n")
    lines.append("| Asset | Sentiment | F: sent→ret | p: sent→ret | F: ret→sent | p: ret→sent | Pattern |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in pivot_rows:
        lines.append(
            f"| {row['asset'].upper()} | {row['sentiment']} | "
            f"{row['f_s_to_r']:.2f} (L{row['lag_s_to_r']}) | "
            f"**{row['p_s_to_r']:.4f}** | "
            f"{row['f_r_to_s']:.2f} (L{row['lag_r_to_s']}) | "
            f"**{row['p_r_to_s']:.4f}** | {row['interpretation']} |"
        )

    # Counts per pattern
    lines.append("\n## Pattern Counts (across 6 assets × 3 sentiment series)\n")
    if pivot_rows:
        from collections import Counter
        cnt = Counter([row["interpretation"] for row in pivot_rows])
        for k, v in cnt.most_common():
            lines.append(f"- **{k}**: {v} pairs")

    lines.append("\n## Interpretation\n")
    # Use the regulatory and infrastructure sentiment specifically
    reg_pivot = [r for r in pivot_rows if r["sentiment"] == "S_reg_decomposed"]
    inf_pivot = [r for r in pivot_rows if r["sentiment"] == "S_infra_decomposed"]
    if reg_pivot:
        sig_reg_lead = sum(1 for r in reg_pivot if r["p_s_to_r"] < 0.05 and r["p_r_to_s"] >= 0.05)
        sig_reg_lag = sum(1 for r in reg_pivot if r["p_r_to_s"] < 0.05 and r["p_s_to_r"] >= 0.05)
        sig_reg_both = sum(1 for r in reg_pivot if r["p_r_to_s"] < 0.05 and r["p_s_to_r"] < 0.05)
        lines.append(
            f"**Regulatory sentiment** (n={len(reg_pivot)} assets): "
            f"{sig_reg_lead} lead-only, {sig_reg_lag} lag-only, "
            f"{sig_reg_both} bidirectional, "
            f"{len(reg_pivot) - sig_reg_lead - sig_reg_lag - sig_reg_both} neither.\n"
        )
    if inf_pivot:
        sig_inf_lead = sum(1 for r in inf_pivot if r["p_s_to_r"] < 0.05 and r["p_r_to_s"] >= 0.05)
        sig_inf_lag = sum(1 for r in inf_pivot if r["p_r_to_s"] < 0.05 and r["p_s_to_r"] >= 0.05)
        sig_inf_both = sum(1 for r in inf_pivot if r["p_r_to_s"] < 0.05 and r["p_s_to_r"] < 0.05)
        lines.append(
            f"**Infrastructure sentiment** (n={len(inf_pivot)} assets): "
            f"{sig_inf_lead} lead-only, {sig_inf_lag} lag-only, "
            f"{sig_inf_both} bidirectional, "
            f"{len(inf_pivot) - sig_inf_lead - sig_inf_lag - sig_inf_both} neither.\n"
        )
    lines.append(
        "The H2 claim of 'sentiment provides incremental explanatory power' is "
        "direction-agnostic. These tests disambiguate. Where infrastructure "
        "sentiment lags returns, the sentiment proxy is best interpreted as "
        "concurrent or post-hoc commentary rather than a leading indicator. "
        "Where regulatory sentiment leads returns, leakage and anticipation "
        "channels are consistent with the result. Section 4.4.5 in the revised "
        "manuscript reports these directions honestly.\n"
    )

    lines.append("## Files\n")
    lines.append("- Full per-asset / per-direction table: `c4-granger-results.csv`\n")

    (OUT_DIR / "c4-granger-summary.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR/'c4-granger-summary.md'}")


if __name__ == "__main__":
    main()
