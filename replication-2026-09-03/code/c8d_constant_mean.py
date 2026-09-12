"""
C8d: Constant-mean justification (Ljung-Box + AR(1)-in-mean robustness).
=======================================================================

The paper uses a constant-mean GJR-GARCH-X (returns are demeaned by the sample
mean before the variance recursion). Reviewer asks whether that is justified or
whether serial correlation in the mean is being dumped into the variance
equation.

(1) Ljung-Box test (statsmodels acorr_ljungbox) on each asset's raw winsorised
    log returns at lags {5, 10, 20}. Report Q-stat and p.

(2) If serial correlation is significant, fit an AR(1)-in-mean variant: regress
    r_t = c + phi*r_{t-1} + e_t (OLS), take the residuals e_t as the mean-filtered
    series, and refit the GJR-GARCH-X on e_t with the SAME event/sentiment exog.
    Compare delta_infra / delta_reg (and the multiplier) to the constant-mean spec.

    Implementation detail: FastTARCHX/TARCHXEstimator both internally demean by the
    SAMPLE MEAN. To make the AR(1) variant actually AR(1)-filtered, we pass the AR(1)
    residuals (already ~zero-mean) as the "returns" series -- the internal sample-mean
    demeaning of an ~zero-mean residual series is a no-op, so the variance recursion
    runs on genuine AR(1) innovations.

Uses the canonical TARCHXEstimator so per-asset deltas carry model SEs/p-values,
matching the headline spec.

Outputs:
    results/c8d-ljungbox.csv
    results/c8d-ar1-vs-constant-mean.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2  # noqa: E402

OUT_DIR = c2.OUT_DIR
DATA_DIR = c2.DATA_DIR
ASSETS = c2.ASSETS
LB_LAGS = [5, 10, 20]


def ar1_filter(returns):
    """OLS AR(1): r_t = c + phi r_{t-1} + e_t. Return residual series e_t (aligned)."""
    r = returns.dropna()
    y = r.iloc[1:].values
    x = r.iloc[:-1].values
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return pd.Series(resid, index=r.index[1:]), beta  # beta = [c, phi]


def fit_canonical(returns, exog):
    est = c2.TARCHXEstimator(returns, exog)
    return est.estimate(method="SLSQP", max_iter=2000)


def main():
    print("Loading panel/sentiment/events...")
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)

    # ---- (1) Ljung-Box ----
    lb_rows = []
    print("\n=== Ljung-Box on raw returns ===")
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)].dropna()
        lb = acorr_ljungbox(r, lags=LB_LAGS, return_df=True)
        for lag in LB_LAGS:
            stat = float(lb.loc[lag, "lb_stat"]); pval = float(lb.loc[lag, "lb_pvalue"])
            lb_rows.append({"asset": a, "lag": lag, "lb_stat": stat, "lb_pvalue": pval,
                            "significant_5pct": pval < 0.05})
        sigs = [f"L{lag}:p={float(lb.loc[lag,'lb_pvalue']):.3f}" for lag in LB_LAGS]
        # AR(1) phi for context
        _, beta = ar1_filter(r)
        print(f"  {a}: {' '.join(sigs)}  | AR(1) phi={beta[1]:+.4f}")
    pd.DataFrame(lb_rows).to_csv(OUT_DIR / "c8d-ljungbox.csv", index=False)

    n_sig = sum(1 for row in lb_rows if row["significant_5pct"])
    print(f"  -> {n_sig}/{len(lb_rows)} (asset x lag) cells significant at 5%")

    # ---- (2) AR(1)-in-mean vs constant-mean ----
    print("\n=== Constant-mean vs AR(1)-in-mean event coefficients ===")
    cmp_rows = []
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)].dropna()

        # constant-mean spec
        dum = c2.build_event_dummies(r.index, inf_d, reg_d,
                                     c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s = sent.reindex(r.index).fillna(0)
        exog = pd.concat([dum[["D_infrastructure", "D_regulatory"]],
                          s[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
                         axis=1).fillna(0)
        res_cm = fit_canonical(r, exog)
        di_cm = res_cm.params.get("D_infrastructure", np.nan)
        dr_cm = res_cm.params.get("D_regulatory", np.nan)
        pi_cm = res_cm.pvalues.get("D_infrastructure", np.nan)
        pr_cm = res_cm.pvalues.get("D_regulatory", np.nan)

        # AR(1)-in-mean spec: filter the mean, refit on residuals
        resid, beta = ar1_filter(r)
        dum2 = c2.build_event_dummies(resid.index, inf_d, reg_d,
                                      c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s2 = sent.reindex(resid.index).fillna(0)
        exog2 = pd.concat([dum2[["D_infrastructure", "D_regulatory"]],
                           s2[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
                          axis=1).fillna(0)
        res_ar = fit_canonical(resid, exog2)
        di_ar = res_ar.params.get("D_infrastructure", np.nan)
        dr_ar = res_ar.params.get("D_regulatory", np.nan)
        pi_ar = res_ar.pvalues.get("D_infrastructure", np.nan)
        pr_ar = res_ar.pvalues.get("D_regulatory", np.nan)

        cmp_rows.append({"asset": a, "ar1_phi": beta[1],
                         "cm_dInfra": di_cm, "cm_dReg": dr_cm, "cm_pInfra": pi_cm, "cm_pReg": pr_cm,
                         "ar1_dInfra": di_ar, "ar1_dReg": dr_ar, "ar1_pInfra": pi_ar, "ar1_pReg": pr_ar})
        print(f"  {a}: CM dInfra={di_cm:.3f} dReg={dr_cm:.3f} | AR1 dInfra={di_ar:.3f} dReg={dr_ar:.3f}")

    df_cmp = pd.DataFrame(cmp_rows)
    df_cmp.to_csv(OUT_DIR / "c8d-ar1-vs-constant-mean.csv", index=False)

    cm_mult = np.nanmean(df_cmp["cm_dInfra"]) / np.nanmean(df_cmp["cm_dReg"])
    ar_mult = np.nanmean(df_cmp["ar1_dInfra"]) / np.nanmean(df_cmp["ar1_dReg"])
    print(f"\nConstant-mean cross-asset multiplier: {cm_mult:.2f}x")
    print(f"AR(1)-in-mean  cross-asset multiplier: {ar_mult:.2f}x")
    print("Saved: c8d-ljungbox.csv, c8d-ar1-vs-constant-mean.csv")


if __name__ == "__main__":
    main()
