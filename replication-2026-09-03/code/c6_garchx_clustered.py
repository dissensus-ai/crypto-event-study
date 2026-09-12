"""
C6: GARCH-X-based pseudoreplication-robust test (the reviewer's "panel GARCH +
clustered SE", done faithfully on the actual estimator rather than a proxy).

Re-fits the GJR-GARCH-X per asset on the baseline 50-event curated sample, extracts
the infrastructure and regulatory event coefficients (delta) and their model standard
errors, then tests the infra-vs-reg difference THREE ways:
  (0) Headline (as in the paper): t-test on the 6 per-asset coefficients as if iid.
  (1) Design-effect correction: inflate the SE by the cross-asset-correlation design
      effect sqrt(1+(N-1)*rho_bar) -- the standard adjustment for averaging correlated units.
  (2) Correlation-weighted combination: Var(mean d) = (1/N^2) sum_ab R_ab se_a se_b,
      using the per-asset model SEs and the cross-asset return-correlation matrix R.
"""
import numpy as np, pandas as pd
from scipy import stats
import c2_relaxed_threshold_sensitivity as c2

panel = c2.load_returns_panel()
common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
sent = c2.load_sentiment_daily(common)
events = pd.read_csv(c2.DATA_DIR/"events.csv"); events["date"]=pd.to_datetime(events["date"])
census = pd.read_csv(c2.OUT_DIR/"c1-dropout-census.csv"); census["date"]=pd.to_datetime(census["date"])
inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)
print(f"baseline events: {len(inf_d)} infra, {len(reg_d)} reg")

di_, dr_, sei_, ser_ = [], [], [], []
for a in c2.ASSETS:
    r = panel[a].loc[panel[a].index>=pd.Timestamp(c2.START_DATE)]
    dum = c2.build_event_dummies(r.index, inf_d, reg_d, c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
    s = sent.reindex(r.index).fillna(0)
    exog = pd.concat([dum[["D_infrastructure","D_regulatory"]],
                      s[["S_gdelt_normalized","S_reg_decomposed","S_infra_decomposed"]]],axis=1).fillna(0)
    res = c2.TARCHXEstimator(r, exog).estimate(method="SLSQP", max_iter=2000)
    di=res.params.get("D_infrastructure",np.nan); dr=res.params.get("D_regulatory",np.nan)
    si=res.std_errors.get("D_infrastructure",np.nan); sr=res.std_errors.get("D_regulatory",np.nan)
    di_.append(di); dr_.append(dr); sei_.append(si); ser_.append(sr)
    print(f"  {a}: dInfra={di:.3f}(se {si:.3f})  dReg={dr:.3f}(se {sr:.3f})")

infra=np.array(di_); reg=np.array(dr_); d=infra-reg
se_d=np.sqrt(np.array(sei_)**2+np.array(ser_)**2)   # per-asset SE of the difference
N=len(d)
# cross-asset return-correlation matrix
R=pd.DataFrame({a:panel[a] for a in c2.ASSETS}).dropna().corr().values
rho_bar=(R.sum()-N)/(N*(N-1))   # mean off-diagonal

# (0) headline iid test
t0,p0=stats.ttest_ind(infra,reg,equal_var=False)
# (1) design-effect on the paired difference
mean_d=d.mean(); se_naive=d.std(ddof=1)/np.sqrt(N)
deff=np.sqrt(1+(N-1)*rho_bar); se_de=se_naive*deff
t1=mean_d/se_de; p1=2*stats.t.sf(abs(t1), df=N-1)
# (2) correlation-weighted using model SEs
var_cw=(R*np.outer(se_d,se_d)).sum()/N**2
t2=mean_d/np.sqrt(var_cw); p2=2*stats.norm.sf(abs(t2))

print("\n=== infra-vs-reg difference, mean d = %.3f, mean cross-asset corr rho_bar = %.2f ===" % (mean_d, rho_bar))
print(f"(0) HEADLINE iid t-test            : t={t0:.3f}  p={p0:.4f}   [paper: t=4.768, p=0.0008]")
print(f"(1) design-effect corrected        : t={t1:.3f}  p={p1:.4f}   (deff={deff:.2f}, eff N={N/deff**2:.1f})")
print(f"(2) correlation-weighted (model SE): t={t2:.3f}  p={p2:.4f}")
print(f"\nmultiplier (mean infra / mean reg) = {infra.mean()/reg.mean():.2f}x")
