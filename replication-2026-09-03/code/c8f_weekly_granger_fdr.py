"""
C8f/C8g: Weekly Granger causality + BH-FDR correction.
======================================================

#9 (c8f). The paper's daily Granger tests forward-fill WEEKLY GDELT sentiment to
daily frequency. Forward-filling a weekly series to daily repeats observations,
changes the effective information content, and makes daily-lag results difficult
to interpret. This script instead tests the association at the series' constructed
weekly frequency. The two analyses answer horizon-specific questions.

This aggregates daily log returns to WEEKLY (sum of daily log returns within each
GDELT week) and weekly ABSOLUTE returns (sum of |daily logret|, a weekly realized-
volatility proxy), aligns them to the NATIVE weekly GDELT sentiment (S_reg_decomposed,
S_infra_decomposed, S_gdelt_normalized) with NO forward-fill, and runs Granger
causality BOTH directions at weekly lags {1..8}. Reports min-p across lags per
(asset, sentiment, direction) and whether sentiment leads weekly volatility,
contrasted against the horizon-mismatched daily-forward-fill result.

#11 (c8g). Benjamini-Hochberg FDR (statsmodels multipletests, method='fdr_bh') on
the family of sentiment->volatility Granger p-values. Applied to:
  (a) the DAILY c4 result family (the version the paper currently reports), and
  (b) the new WEEKLY family.
Reports how many of the currently-"significant" (p<0.05) relationships survive at
q<0.05 and q<0.10.

Weekly sentiment is built directly from data/gdelt.csv (week_start), recomputing
S_gdelt_normalized with the same 52-week rolling z (min_periods=26) and the
decomposed series as normalized * proportion -- matching the pipeline, but kept
weekly (no daily reindex/ffill).

Outputs:
    results/c8f-weekly-granger.csv
    results/c8g-fdr-granger.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2  # noqa: E402

OUT_DIR = c2.OUT_DIR
DATA_DIR = c2.DATA_DIR
ASSETS = c2.ASSETS
SENT_COLS = ["S_reg_decomposed", "S_infra_decomposed", "S_gdelt_normalized"]
WEEKLY_MAXLAG = 8


def load_weekly_sentiment():
    """Native weekly GDELT sentiment -- NO daily reindex / ffill."""
    df = pd.read_csv(DATA_DIR / "gdelt.csv")
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True).dt.tz_convert(None).dt.normalize()
    df = df.set_index("week_start").sort_index()
    win, mp = 52, 26
    rmean = df["S_gdelt_raw"].rolling(win, min_periods=mp).mean()
    rstd = df["S_gdelt_raw"].rolling(win, min_periods=mp).std()
    df["S_gdelt_normalized"] = (df["S_gdelt_raw"] - rmean) / rstd
    df["S_reg_decomposed"] = df["S_gdelt_normalized"] * df["reg_proportion"]
    df["S_infra_decomposed"] = df["S_gdelt_normalized"] * df["infra_proportion"]
    out = df[SENT_COLS].copy()
    # honour the pipeline's pre-2019-06 zeroing convention
    out.loc[out.index < c2.SENTIMENT_CUTOFF] = 0.0
    return out


def weekly_returns(daily_ret, week_index):
    """Aggregate daily log returns to the GDELT weekly grid.

    week k covers [week_start_k, week_start_{k+1}). signed = sum of daily logret;
    vol proxy = sum of |daily logret| (weekly realized volatility).
    """
    we = list(week_index) + [week_index.max() + pd.Timedelta(days=7)]
    signed, absvol = [], []
    for i in range(len(week_index)):
        lo, hi = we[i], we[i + 1]
        chunk = daily_ret.loc[(daily_ret.index >= lo) & (daily_ret.index < hi)]
        signed.append(chunk.sum() if len(chunk) else np.nan)
        absvol.append(chunk.abs().sum() if len(chunk) else np.nan)
    return (pd.Series(signed, index=week_index, name="wk_ret"),
            pd.Series(absvol, index=week_index, name="wk_absvol"))


def granger_minp(y, x, maxlag):
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 40:
        return np.nan, np.nan, np.nan, len(df)
    df.columns = ["y", "x"]
    try:
        res = grangercausalitytests(df[["y", "x"]], maxlag=maxlag, verbose=False)
    except Exception:
        return np.nan, np.nan, np.nan, len(df)
    pvals, fstats = {}, {}
    for lag in range(1, maxlag + 1):
        if lag in res:
            f, p, _, _ = res[lag][0]["ssr_ftest"]
            pvals[lag] = float(p); fstats[lag] = float(f)
    if not pvals:
        return np.nan, np.nan, np.nan, len(df)
    opt = min(pvals, key=pvals.get)
    return fstats[opt], pvals[opt], opt, len(df)


def main():
    print("Loading daily panel + native weekly sentiment...")
    panel = c2.load_returns_panel()
    wsent = load_weekly_sentiment()
    week_index = wsent.index

    # ---------- #9 weekly Granger ----------
    rows = []
    for a in ASSETS:
        dr = panel[a].dropna()
        wk_ret, wk_absvol = weekly_returns(dr, week_index)
        for sc in SENT_COLS:
            s = wsent[sc]
            # use abs-vol as the volatility series (matches |returns| in c4)
            f_sr, p_sr, l_sr, n_sr = granger_minp(wk_absvol, s, WEEKLY_MAXLAG)   # sent->vol
            f_rs, p_rs, l_rs, n_rs = granger_minp(s, wk_absvol, WEEKLY_MAXLAG)   # vol->sent
            rows.append({"asset": a, "sentiment": sc, "freq": "weekly",
                         "f_sent_to_vol": f_sr, "p_sent_to_vol": p_sr, "lag_sent_to_vol": l_sr,
                         "f_vol_to_sent": f_rs, "p_vol_to_sent": p_rs, "lag_vol_to_sent": l_rs,
                         "n_weeks": n_sr})
            print(f"  {a} {sc:20s}: sent->vol p={p_sr:.3f}(L{l_sr}) | vol->sent p={p_rs:.3f}(L{l_rs}) [n={n_sr}]")
    wk = pd.DataFrame(rows)
    wk.to_csv(OUT_DIR / "c8f-weekly-granger.csv", index=False)

    n_lead_wk = ((wk["p_sent_to_vol"] < 0.05) & (wk["p_vol_to_sent"] >= 0.05)).sum()
    n_sig_sr_wk = (wk["p_sent_to_vol"] < 0.05).sum()
    print(f"\n  WEEKLY: {n_sig_sr_wk}/{len(wk)} sent->vol significant (p<0.05); "
          f"{n_lead_wk} sentiment-LEADS-only.")

    # ---------- #11 BH-FDR ----------
    fdr_rows = []

    # (a) daily family from c4
    c4 = pd.read_csv(OUT_DIR / "c4-granger-results.csv")
    daily_sr = c4[c4["direction"] == "sentiment_to_returns"].copy()
    p_daily = daily_sr["p_value_min"].values
    for q in [0.05, 0.10]:
        rej, padj, *_ = multipletests(p_daily, alpha=q, method="fdr_bh")
        fdr_rows.append({"family": "daily_ffill_sent_to_vol", "q": q, "n_tests": len(p_daily),
                         "n_raw_sig_0.05": int((p_daily < 0.05).sum()),
                         "n_survive_fdr": int(rej.sum())})
    # also the full 36-test daily family (both directions), for completeness
    p_daily_all = c4["p_value_min"].values
    for q in [0.05, 0.10]:
        rej, *_ = multipletests(p_daily_all, alpha=q, method="fdr_bh")
        fdr_rows.append({"family": "daily_ffill_BOTH_directions", "q": q, "n_tests": len(p_daily_all),
                         "n_raw_sig_0.05": int((p_daily_all < 0.05).sum()),
                         "n_survive_fdr": int(rej.sum())})

    # (b) weekly family
    p_wk_sr = wk["p_sent_to_vol"].dropna().values
    for q in [0.05, 0.10]:
        rej, *_ = multipletests(p_wk_sr, alpha=q, method="fdr_bh")
        fdr_rows.append({"family": "weekly_sent_to_vol", "q": q, "n_tests": len(p_wk_sr),
                         "n_raw_sig_0.05": int((p_wk_sr < 0.05).sum()),
                         "n_survive_fdr": int(rej.sum())})
    p_wk_all = pd.concat([wk["p_sent_to_vol"], wk["p_vol_to_sent"]]).dropna().values
    for q in [0.05, 0.10]:
        rej, *_ = multipletests(p_wk_all, alpha=q, method="fdr_bh")
        fdr_rows.append({"family": "weekly_BOTH_directions", "q": q, "n_tests": len(p_wk_all),
                         "n_raw_sig_0.05": int((p_wk_all < 0.05).sum()),
                         "n_survive_fdr": int(rej.sum())})

    df_fdr = pd.DataFrame(fdr_rows)
    df_fdr.to_csv(OUT_DIR / "c8g-fdr-granger.csv", index=False)
    print("\n=== BH-FDR ===")
    for _, rr in df_fdr.iterrows():
        print(f"  {rr['family']:32s} q={rr['q']:.2f}: raw_sig(0.05)={rr['n_raw_sig_0.05']}/{rr['n_tests']} "
              f"-> survive FDR = {rr['n_survive_fdr']}")
    print("Saved: c8f-weekly-granger.csv, c8g-fdr-granger.csv")


if __name__ == "__main__":
    main()
