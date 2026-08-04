"""
C15: Does the weekly sentiment->volatility lead survive controlling for the
     known event shocks? (Referee point: uncontrolled event confounding in VAR)
============================================================================

A reviewer objects that the weekly bivariate Granger VAR (sentiment -> realised
volatility) omits the 50 classified events as exogenous controls. Because an
event mechanically spikes BOTH GDELT sentiment and realised volatility, an
unmodelled event can manufacture a spurious sentiment->volatility lead: sentiment
rises the week of the event, volatility stays elevated the following week(s)
through GARCH persistence, and the naive test reads the sentiment rise as
"leading" the volatility it in fact merely co-moves with.

This script re-tests the lead with weekly-aggregated event dummies added to the
VAR as EXOGENOUS controls, reusing the c12 conditional-Granger SSR-F machinery
(same design already used there for the Bitcoin-co-movement and litigation-window
controls). The event dummies enter CONTEMPORANEOUSLY and at lags 1..p, so the
control absorbs both the same-week shock and its persistence into the following
weeks. If sentiment's predictive content is a pure event shadow, the sentiment
block should lose significance once the event block is in the regression.

Event weekly aggregation: each of the 50 events is mapped to the GDELT week
[week_start, week_start + 7) that contains it. We build three weekly control
series from events.csv:
    ev_all   = count of ANY event in the week
    ev_infra = count of Infrastructure events in the week
    ev_reg   = count of Regulatory events in the week
Two control specifications are reported:
    (1) TOTAL   : single ev_all control (contemp + lags)
    (2) SPLIT   : ev_infra AND ev_reg controls together (contemp + lags)
The SPLIT spec is the strict reading of "include the events as controls" because
it lets infrastructure and regulatory shocks load differently on volatility.

Outputs (event-study/r1-revision/):
    c15-granger-event-dummies.csv       (per (asset,sentiment) pair, both specs)
    c15-granger-event-dummies-summary.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2   # noqa: E402
import c8f_weekly_granger_fdr as c8f            # noqa: E402
import c12_granger_rigour as c12                # noqa: E402

OUT_DIR = c2.OUT_DIR
DATA_DIR = c2.DATA_DIR
ASSETS = c2.ASSETS
SENT_COLS = ["S_reg_decomposed", "S_infra_decomposed", "S_gdelt_normalized"]
WEEKLY_MAXLAG = 8
ALPHA = 0.05


# ----------------------------------------------------------------------------
# Weekly event-dummy construction on the GDELT week grid.
# ----------------------------------------------------------------------------
def build_weekly_event_dummies(week_index):
    """Map the 50 events to GDELT weeks and return weekly count series.

    week k spans [week_index[k], week_index[k] + 7 days). Returns a DataFrame
    indexed by week_index with columns ev_all, ev_infra, ev_reg.
    """
    ev = pd.read_csv(DATA_DIR / "events.csv")
    ev["date"] = pd.to_datetime(ev["date"])
    wk = pd.DatetimeIndex(week_index)
    edges = list(wk) + [wk.max() + pd.Timedelta(days=7)]
    ev_all = np.zeros(len(wk))
    ev_infra = np.zeros(len(wk))
    ev_reg = np.zeros(len(wk))
    for _, row in ev.iterrows():
        d = row["date"]
        # locate the containing week
        for i in range(len(wk)):
            if edges[i] <= d < edges[i + 1]:
                ev_all[i] += 1
                if str(row["type"]).lower().startswith("infra"):
                    ev_infra[i] += 1
                elif str(row["type"]).lower().startswith("reg"):
                    ev_reg[i] += 1
                break
    out = pd.DataFrame(
        {"ev_all": ev_all, "ev_infra": ev_infra, "ev_reg": ev_reg},
        index=week_index,
    )
    return out


# ----------------------------------------------------------------------------
# Conditional Granger with contemporaneous + lagged exogenous controls.
# Mirrors c12.conditional_granger but (a) accepts several control series and
# (b) includes the CONTEMPORANEOUS control term z_t as well as lags z_{t-1..t-p},
# so a same-week event shock to volatility is absorbed, not just its lags.
# ----------------------------------------------------------------------------
def conditional_granger_events(y, x, controls, maxlag=WEEKLY_MAXLAG):
    """SSR-F test of x -> y at each lag p, controlling for contemporaneous and
    lagged values of each control series in `controls` (a list of pd.Series).

    Unrestricted: y_t ~ c + sum_{1..p} y + sum_{1..p} x
                        + [for each ctrl c: c_t + sum_{1..p} c]
    Restricted:   drop the x lags. F on the x block. min-p across p=1..maxlag.
    """
    frame = pd.concat([y.rename("y"), x.rename("x")] +
                      [c.rename(f"z{i}") for i, c in enumerate(controls)], axis=1).dropna()
    if len(frame) < 50:
        return np.nan, np.nan, np.nan, len(frame)
    zcols_base = [f"z{i}" for i in range(len(controls))]
    best_p, best_f, best_lag = np.nan, np.nan, np.nan
    for p in range(1, maxlag + 1):
        d = frame.copy()
        cols_y = [f"y_l{j}" for j in range(1, p + 1)]
        cols_x = [f"x_l{j}" for j in range(1, p + 1)]
        for j in range(1, p + 1):
            d[f"y_l{j}"] = d["y"].shift(j)
            d[f"x_l{j}"] = d["x"].shift(j)
        ctrl_cols = []
        for zc in zcols_base:
            ctrl_cols.append(zc)  # contemporaneous
            for j in range(1, p + 1):
                col = f"{zc}_l{j}"
                d[col] = d[zc].shift(j)
                ctrl_cols.append(col)
        d = d.dropna()
        k_un = 1 + len(cols_y) + len(cols_x) + len(ctrl_cols)
        if len(d) < (k_un + 5):
            continue
        Y = d["y"].values
        X_un = sm.add_constant(d[cols_y + cols_x + ctrl_cols].values)
        X_re = sm.add_constant(d[cols_y + ctrl_cols].values)
        try:
            m_un = sm.OLS(Y, X_un).fit()
            m_re = sm.OLS(Y, X_re).fit()
        except Exception:
            continue
        ssr_un, ssr_re = m_un.ssr, m_re.ssr
        q = p
        df_den = len(d) - X_un.shape[1]
        if df_den <= 0:
            continue
        F = ((ssr_re - ssr_un) / q) / (ssr_un / df_den)
        pval = float(sps.f.sf(F, q, df_den))
        if np.isnan(best_p) or pval < best_p:
            best_p, best_f, best_lag = pval, F, p
    return best_f, best_p, best_lag, len(frame)


def main():
    print("=" * 78)
    print("C15: sentiment -> volatility lead, controlling for event dummies")
    print("=" * 78)
    voldf, wsent, week_index = c12.build_weekly_panel()
    evd = build_weekly_event_dummies(week_index)
    print(f"weekly grid: {len(week_index)} weeks; events mapped: "
          f"all={int(evd['ev_all'].sum())}, infra={int(evd['ev_infra'].sum())}, "
          f"reg={int(evd['ev_reg'].sum())}")

    # FDR-7 survivors baseline (from c8f) for tallying
    base = pd.read_csv(OUT_DIR / "c8f-weekly-granger.csv")
    p_base = base["p_sent_to_vol"].values
    rej_base, _, *_ = multipletests(p_base, alpha=0.05, method="fdr_bh")
    base["fdr_q005"] = rej_base
    fdr7 = set(base[base["fdr_q005"]][["asset", "sentiment"]].apply(tuple, axis=1))

    rows = []
    for a in ASSETS:
        for sc in SENT_COLS:
            y = voldf[a]; x = wsent[sc]
            f0, p0, lag0, n0 = c8f.granger_minp(y, x, WEEKLY_MAXLAG)
            fT, pT, lagT, nT = conditional_granger_events(
                y, x, [evd["ev_all"]], WEEKLY_MAXLAG)
            fS, pS, lagS, nS = conditional_granger_events(
                y, x, [evd["ev_infra"], evd["ev_reg"]], WEEKLY_MAXLAG)
            rows.append({
                "asset": a, "sentiment": sc,
                "p_uncontrolled": p0, "lag_uncontrolled": lag0,
                "p_ctrl_total": pT, "lag_ctrl_total": lagT,
                "p_ctrl_split": pS, "lag_ctrl_split": lagS,
                "n_weeks": nT,
                "is_fdr7": (a, sc) in fdr7,
                "unctrl_sig": (p0 < ALPHA) if pd.notna(p0) else False,
                "total_sig": (pT < ALPHA) if pd.notna(pT) else False,
                "split_sig": (pS < ALPHA) if pd.notna(pS) else False,
            })
    df = pd.DataFrame(rows)

    # BH-FDR on the controlled p-values across the 18-pair family
    for col, tag in [("p_ctrl_total", "total"), ("p_ctrl_split", "split")]:
        pv = df[col].values
        valid = pd.notna(pv)
        df[f"{tag}_fdr_q005"] = False
        df[f"{tag}_fdr_q010"] = False
        if valid.sum() > 0:
            r05, _, *_ = multipletests(pv[valid], alpha=0.05, method="fdr_bh")
            r10, _, *_ = multipletests(pv[valid], alpha=0.10, method="fdr_bh")
            df.loc[valid, f"{tag}_fdr_q005"] = r05
            df.loc[valid, f"{tag}_fdr_q010"] = r10
    df.to_csv(OUT_DIR / "c15-granger-event-dummies.csv", index=False)

    print("\nPer-pair (uncontrolled -> event-controlled):")
    for _, r in df.iterrows():
        star = "*FDR7*" if r["is_fdr7"] else "      "
        print(f"  {star} {r['asset']} {r['sentiment']:20s} "
              f"uncond p={r['p_uncontrolled']:.4f} | "
              f"total-ctrl p={r['p_ctrl_total']:.4f} | "
              f"split-ctrl p={r['p_ctrl_split']:.4f}")

    # tallies
    n_raw = int(df["unctrl_sig"].sum())
    n_fdr7 = len(fdr7)
    total_survivors = set(df[df["total_sig"]][["asset", "sentiment"]].apply(tuple, axis=1))
    split_survivors = set(df[df["split_sig"]][["asset", "sentiment"]].apply(tuple, axis=1))
    n_total_of7 = len(fdr7 & total_survivors)
    n_split_of7 = len(fdr7 & split_survivors)
    n_total_fdr = int(df["total_fdr_q005"].sum())
    n_split_fdr = int(df["split_fdr_q005"].sum())

    summary = pd.DataFrame([
        {"spec": "uncontrolled raw p<0.05 (of 18)", "n": n_raw},
        {"spec": "baseline BH-FDR q<0.05 survivors", "n": n_fdr7},
        {"spec": "TOTAL event-ctrl raw p<0.05 (of 18)", "n": int(df["total_sig"].sum())},
        {"spec": "TOTAL event-ctrl: FDR-7 retained", "n": n_total_of7},
        {"spec": "TOTAL event-ctrl BH-FDR q<0.05 (of 18)", "n": n_total_fdr},
        {"spec": "SPLIT event-ctrl raw p<0.05 (of 18)", "n": int(df["split_sig"].sum())},
        {"spec": "SPLIT event-ctrl: FDR-7 retained", "n": n_split_of7},
        {"spec": "SPLIT event-ctrl BH-FDR q<0.05 (of 18)", "n": n_split_fdr},
    ])
    summary.to_csv(OUT_DIR / "c15-granger-event-dummies-summary.csv", index=False)
    print("\n" + "=" * 78)
    print("BOTTOM LINE")
    print("=" * 78)
    for _, r in summary.iterrows():
        print(f"  {r['spec']:44s} {r['n']}")
    print(f"\n  FDR-7 baseline survivors: {sorted(fdr7)}")
    print(f"  Of those, retained under TOTAL event control: {n_total_of7}/7")
    print(f"  Of those, retained under SPLIT event control: {n_split_of7}/7")
    print("\nSaved: c15-granger-event-dummies.csv, c15-granger-event-dummies-summary.csv")


if __name__ == "__main__":
    main()
