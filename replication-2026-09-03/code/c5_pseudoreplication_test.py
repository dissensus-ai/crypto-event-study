"""
C5: Pseudoreplication-robust re-test of the infrastructure-vs-regulatory asymmetry.

The headline test (t=4.768, p=0.0008) treats the 6 per-asset GJR-GARCH-X event
coefficients as N=6 independent observations. They are not: the assets share the
same classified 50-event calendar and correlate 0.54-0.83 (BNB's later listing
leaves 45 supported event windows). This re-tests the asymmetry with the EVENT
as the unit of analysis, two ways:
  (A) Event-level (N=50): average the abnormal-variance response across assets within
      each event -> one value per event; test infra (26) vs reg (24).
  (B) Cluster-robust panel: asset x event abnormal-variance regressed on an
      infrastructure dummy, SE clustered by EVENT (50 clusters).
Abnormal variance proxy: mean squared daily log-return (in %^2) in the [-3,+3] event
window minus the asset's full-sample mean squared return (model-free, matches the
variance units of the GJR-GARCH-X delta coefficients).
"""
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # legacy event-study/ working-tree fallback
DATA = HERE / "data" if (HERE / "data").is_dir() else ROOT / "code" / "data"
ASSETS = ["btc","eth","xrp","bnb","ltc","ada"]
W_BEFORE, W_AFTER = 3, 3

def load_ret(a):
    df = pd.read_csv(DATA/f"{a}.csv")
    df["date"] = pd.to_datetime(df["snapped_at"], utc=True).dt.tz_convert(None).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date").set_index("date").loc["2019-01-01":"2025-08-31"]
    return (np.log(df["price"]).diff()*100).dropna()

panel = {a: load_ret(a) for a in ASSETS}
sq = {a: panel[a]**2 for a in ASSETS}             # squared returns (variance proxy, %^2)
base = {a: sq[a].mean() for a in ASSETS}          # full-sample mean squared return

ev = pd.read_csv(DATA/"events.csv"); ev["date"]=pd.to_datetime(ev["date"])
ev = ev[ev["type"].isin(["Infrastructure","Regulatory"])].reset_index(drop=True)
print(f"events: {ev['type'].value_counts().to_dict()}")

rows=[]
for i,e in ev.iterrows():
    d=e["date"]; lo=d-pd.Timedelta(days=W_BEFORE); hi=d+pd.Timedelta(days=W_AFTER)
    for a in ASSETS:
        s=sq[a]; win=s[(s.index>=lo)&(s.index<=hi)]
        if len(win)>0:
            rows.append({"event":i,"type":e["type"],"asset":a,
                         "abn_var": win.mean()-base[a]})
df=pd.DataFrame(rows)
df["infra"]=(df["type"]=="Infrastructure").astype(int)

# (A) event-level: mean across assets per event
evl=df.groupby(["event","type"])["abn_var"].mean().reset_index()
inf=evl.loc[evl["type"]=="Infrastructure","abn_var"]; reg=evl.loc[evl["type"]=="Regulatory","abn_var"]
t,p=stats.ttest_ind(inf,reg,equal_var=False)
U,pu=stats.mannwhitneyu(inf,reg,alternative="two-sided")
print("\n=== (A) EVENT-LEVEL (N=50; events as unit) ===")
print(f"  infra mean abn-var = {inf.mean():.3f} (n={len(inf)}); reg mean = {reg.mean():.3f} (n={len(reg)})")
print(f"  ratio infra/reg    = {inf.mean()/reg.mean():.2f}x" if reg.mean()>0 else "  ratio n/a")
print(f"  Welch t = {t:.3f}, p = {p:.4f}")
print(f"  Mann-Whitney U = {U:.0f}, p = {pu:.4f}")

# (B) cluster-robust panel: abn_var ~ infra, cluster by event
try:
    import statsmodels.formula.api as smf
    m=smf.ols("abn_var ~ infra", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["event"]})
    b=m.params["infra"]; se=m.bse["infra"]; tt=m.tvalues["infra"]; pp=m.pvalues["infra"]
    print("\n=== (B) CLUSTER-ROBUST PANEL (asset x event = %d obs; SE clustered by event, %d clusters) ===" % (len(df), df["event"].nunique()))
    print(f"  infra coefficient = {b:.3f}  (cluster-robust SE = {se:.3f})")
    print(f"  t = {tt:.3f}, p = {pp:.4f}")
except Exception as e:
    print("\n[B] statsmodels unavailable:", e)

print("\n=== HEADLINE (current paper, N=6 cross-asset coefficients) ===")
print("  t = 4.768, p = 0.0008  (treats 6 correlated assets as independent)")
