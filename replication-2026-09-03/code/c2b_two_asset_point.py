"""
C2b: the MISSING like-for-like 2-asset point on the 135 reconstructed pool.

Table 7 (c2-summary-table.csv) reports S1=legacy-50, S2=relaxed(115),
S3=nofilter(135), S4=strict(77) -- but NOT the 2-asset Stage-2 screen on the
135 pool (stage2_std_pass, n=93). The reframed "multiplier peaks at the
2-asset threshold" claim leans on S1 (the legacy 50-event hand-curated
sample), which is not a 2-asset re-screen of the pool and breaks the nesting
(50 < 77). This script computes the genuine 2-asset point under the SAME
global-clip pipeline as S2-S4, so the comparison is apples-to-apples.
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import c2_relaxed_threshold_sensitivity as c2  # reuses loaders + estimator

def run():
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sentiment = c2.load_sentiment_daily(common)

    census = pd.read_csv(c2.OUT_DIR / "c1-dropout-census.csv")
    census["date"] = pd.to_datetime(census["date"])
    sub = census.loc[census["stage2_std_pass"].astype(bool)]
    inf_dates = sub.loc[sub["tentative_category"] == "Infrastructure", "date"].tolist()
    reg_dates = sub.loc[sub["tentative_category"] == "Regulatory", "date"].tolist()
    print(f"=== S_2asset (stage2_std_pass on 135 pool): {len(inf_dates)} infra, {len(reg_dates)} reg ===")

    infra_coefs, reg_coefs = [], []
    for a in c2.ASSETS:
        returns = panel[a].copy()
        returns = returns.loc[returns.index >= pd.Timestamp(c2.START_DATE)]
        dummies = c2.build_event_dummies(returns.index, inf_dates, reg_dates,
                                         window_before=c2.WINDOW_DAYS_BEFORE,
                                         window_after=c2.WINDOW_DAYS_AFTER)
        sent = sentiment.reindex(returns.index).fillna(0)
        exog = pd.concat([dummies[["D_infrastructure", "D_regulatory"]],
                          sent[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
                         axis=1).fillna(0)
        est = c2.TARCHXEstimator(returns, exog)
        res = est.estimate(method="SLSQP", max_iter=2000)
        di = res.params.get("D_infrastructure", np.nan)
        dr = res.params.get("D_regulatory", np.nan)
        flag = "" if res.converged else "  [WARN not converged]"
        print(f"  {a}: dInfra={di:.4f}  dReg={dr:.4f}{flag}")
        infra_coefs.append(di); reg_coefs.append(dr)

    ia = np.array(infra_coefs, float); ra = np.array(reg_coefs, float)
    inf_mean, reg_mean = np.nanmean(ia), np.nanmean(ra)
    mult = inf_mean / reg_mean if reg_mean != 0 else np.nan
    t, p = ttest_ind(ia, ra, equal_var=False, nan_policy="omit")
    sp = np.sqrt((np.nanvar(ia, ddof=1) + np.nanvar(ra, ddof=1)) / 2)
    d = (inf_mean - reg_mean) / sp if sp > 0 else np.nan

    print(f"\n--- 2-asset (n={len(sub)}) cross-asset result ---")
    print(f"  mean infra  = {inf_mean:.4f}")
    print(f"  mean reg    = {reg_mean:.4f}")
    print(f"  MULTIPLIER  = {mult:.3f}x")
    print(f"  Welch t={t:.3f}  p={p:.4f}   Cohen d={d:.3f}")
    print("\n--- context (from c2-summary-table.csv, same pipeline) ---")
    summary = pd.read_csv(c2.OUT_DIR / "c2-summary-table.csv")
    for _, row in summary.iterrows():
        print(f"  {row['spec']:<18} = {row['multiplier']:.3f}x (p={row['welch_p']:.4f})")
    print(f"  >>> S_2asset-{len(sub)}  = {mult:.3f}x (p={p:.4f})  <-- the like-for-like point")

    out = pd.DataFrame([{
        "spec": "S2_twoasset",
        "n_infra_events": len(inf_dates),
        "n_reg_events": len(reg_dates),
        "mean_infra_coef": inf_mean,
        "mean_reg_coef": reg_mean,
        "multiplier": mult,
        "cohens_d": d,
        "welch_t": t,
        "welch_p": p,
    }])
    out_path = c2.OUT_DIR / "c2b-two-asset-result.csv"
    out.to_csv(out_path, index=False)
    print(f"  saved {out_path}")
    return out.iloc[0].to_dict()

if __name__ == "__main__":
    run()
