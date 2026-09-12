"""
C12: Granger rigour — does the one positive result survive proper inference?
============================================================================

The paper's single positive finding is that *weekly* GDELT sentiment Granger-LEADS
weekly volatility: in the native-weekly test (c8f), 10/18 (asset x sentiment)
sent->vol pairs are raw-significant at p<0.05 and 7 survive BH-FDR at q<0.05
(c8g). The paper currently frames this as a "genuine anticipatory channel".

This script stress-tests that claim with five corrections, on the SAME native
weekly data (6 assets, native weekly GDELT, no forward-fill), reusing the c8f
weekly machinery (load_weekly_sentiment, weekly_returns, granger_minp) and the
c2 returns panel.

  #D2 Stationarity:  ADF + KPSS on each weekly Sigma|r| vol proxy and both GDELT
       sentiment series (reg, infra) + the normalized series. I(0)/I(1) verdict.

  #D2 Toda-Yamamoto: lag-augmented VAR (p by AIC/BIC + d_max extra lags; Wald on
       the first p lags only). The naive grangercausalitytests F-test is INVALID
       if either series is I(1) (spurious-regression / non-standard asymptotics).
       Report TY chi2 p vs naive F p for each pair.

  #D3 Bitcoin common-driver control: re-run sent->vol adding lagged BTC weekly
       vol (Sigma|r_BTC|) as an exogenous control in each altcoin's VAR. For BTC
       and ETH (no exogenous BTC of their own / collinear) use an
       aggregate-market vol control = mean Sigma|r| over the OTHER 5 assets.
       Does the lead survive conditioning on BTC/market co-movement?

  #C4 Missingness: the normalized weekly sentiment has a 25-week initialization
       warm-up. The first-lag design leaves 24 warm-up observations for assets
       with full return histories and none for later-starting BNB. Logistic
       regression / chi-square of this warm-up indicator on lagged weekly
       volatility tests whether initialization could manufacture a spurious lead.

  #E1 Litigation confound: the 7 FDR survivors are XRP, ADA, BNB — all SEC-
       litigation assets. Re-run Granger for those three (a) on the PRE-litigation
       subperiod and (b) with a litigation-active exogenous indicator in the VAR.
       Do the leads survive outside / net of litigation windows?

Outputs (all in results/):
    c12-granger-rigour-stationarity.csv
    c12-granger-rigour-toda-yamamoto.csv
    c12-granger-rigour-btc-control.csv
    c12-granger-rigour-missingness.csv
    c12-granger-rigour-litigation.csv
    c12-granger-rigour-bottomline.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from itertools import product

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy import stats as sps

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2  # noqa: E402
import c8f_weekly_granger_fdr as c8f  # noqa: E402

OUT_DIR = c2.OUT_DIR
ASSETS = c2.ASSETS
SENT_COLS = ["S_reg_decomposed", "S_infra_decomposed", "S_gdelt_normalized"]
WEEKLY_MAXLAG = 8

# Litigation onset dates (SEC actions naming each asset). XRP: SEC v Ripple.
# BNB: SEC v Binance. ADA: named as alleged unregistered security in
# SEC v Binance AND SEC v Coinbase complaints (both Jun 2023).
LITIGATION_ONSET = {
    "xrp": pd.Timestamp("2020-12-22"),  # SEC v Ripple
    "bnb": pd.Timestamp("2023-06-05"),  # SEC v Binance
    "ada": pd.Timestamp("2023-06-05"),  # SEC v Binance/Coinbase complaints name ADA
}
# Litigation conclusion (re-clarified) for XRP — used only to bound the active window.
LITIGATION_END = {
    "xrp": pd.Timestamp("2025-08-07"),  # XRP case concludes (SEC LR-26369, UTC date;
                                        # same GDELT week as the pre-correction 08-08,
                                        # so the weekly masks are unchanged)
    "bnb": pd.Timestamp("2025-12-31"),  # ongoing through sample end
    "ada": pd.Timestamp("2025-12-31"),
}

ALPHA = 0.05


# ----------------------------------------------------------------------------
# Shared: assemble weekly panel (vol per asset + sentiment), reusing c8f
# ----------------------------------------------------------------------------
def build_weekly_panel():
    panel = c2.load_returns_panel()
    wsent = c8f.load_weekly_sentiment()
    week_index = wsent.index
    vol = {}
    for a in ASSETS:
        dr = panel[a].dropna()
        _, wk_absvol = c8f.weekly_returns(dr, week_index)
        vol[a] = wk_absvol
    voldf = pd.DataFrame(vol)
    voldf.index = week_index
    return voldf, wsent, week_index


# ----------------------------------------------------------------------------
# #D2a Stationarity
# ----------------------------------------------------------------------------
def stationarity_verdict(s, name):
    s = s.dropna()
    n = len(s)
    out = {"series": name, "n": n}
    try:
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        out["adf_stat"] = adf_stat
        out["adf_p"] = adf_p
        out["adf_rej_unitroot_005"] = adf_p < 0.05  # reject H0(unit root) => stationary
    except Exception as e:
        out["adf_p"] = np.nan; out["adf_rej_unitroot_005"] = np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        out["kpss_stat"] = kpss_stat
        out["kpss_p"] = kpss_p
        out["kpss_rej_stationary_005"] = kpss_p < 0.05  # reject H0(stationary) => nonstationary
    except Exception as e:
        out["kpss_p"] = np.nan; out["kpss_rej_stationary_005"] = np.nan
    # joint verdict
    adf_stat_ok = out.get("adf_rej_unitroot_005")
    kpss_nonstat = out.get("kpss_rej_stationary_005")
    if adf_stat_ok and not kpss_nonstat:
        v = "I(0)"
    elif (not adf_stat_ok) and kpss_nonstat:
        v = "I(1)"
    elif adf_stat_ok and kpss_nonstat:
        v = "ambiguous/near-I(1)"  # both reject — fractional / structural
    else:
        v = "ambiguous (neither rejects)"
    out["verdict"] = v
    return out


def run_stationarity(voldf, wsent):
    rows = []
    for a in ASSETS:
        rows.append(stationarity_verdict(voldf[a], f"vol_{a}"))
    for sc in SENT_COLS:
        rows.append(stationarity_verdict(wsent[sc], sc))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c12-granger-rigour-stationarity.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# #D2b Toda-Yamamoto: lag-augmented VAR, Wald on first p lags of x in y-eqn.
# x -> y means: does x Granger-cause y. We test sent -> vol, so y=vol, x=sent.
# ----------------------------------------------------------------------------
def toda_yamamoto(y, x, maxlag=WEEKLY_MAXLAG, d_max=1):
    """Toda-Yamamoto (1995) lag-augmented Granger test for x Granger-causing y.

    Implemented as a SINGLE-EQUATION OLS to avoid the VAR cov_params stacking
    ambiguity (the earlier VAR-cov_params restriction gave wrong Wald stats):
      1. Pick VAR order p by AIC (capped at maxlag) on the bivariate [y, x].
      2. Estimate y_t = c + sum_{j=1..k} a_j y_{t-j} + sum_{j=1..k} b_j x_{t-j}
         with k = p + d_max (d_max extra "augmentation" lags; d_max = max
         suspected integration order, here 1 since series are I(0)/near-I(1)).
      3. HC1-robust Wald test that the FIRST p x-coefficients (b_1..b_p) = 0.
         The extra d_max x-lags are left UNRESTRICTED -- this is what restores
         the standard chi^2(p) asymptotics even if a series has a unit root.
    Validated: built-in VARResults.test_causality(0,[1]) reproduces the
    full-lag version (e.g. ltc/S_infra chi2=1.76, matching the non-significant
    naive p), confirming the OLS restriction is correct.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 50:
        return np.nan, np.nan, np.nan, np.nan, len(df)
    df.columns = ["y", "x"]
    data = df[["y", "x"]].values
    # 1. order selection
    try:
        sel = VAR(data).select_order(maxlags=min(maxlag, len(df) // 5))
        p = int(sel.aic) if sel.aic and sel.aic >= 1 else 1
    except Exception:
        p = 1
    p = max(p, 1)
    k = p + d_max
    # 2. lag-augmented single-equation design
    d = df.copy()
    ylags = [f"y_l{j}" for j in range(1, k + 1)]
    xlags = [f"x_l{j}" for j in range(1, k + 1)]
    for j in range(1, k + 1):
        d[f"y_l{j}"] = d["y"].shift(j)
        d[f"x_l{j}"] = d["x"].shift(j)
    d = d.dropna()
    if len(d) < (2 * k + 5):
        return np.nan, np.nan, p, k, len(d)
    Y = d["y"].values
    X = sm.add_constant(d[ylags + xlags].values)
    try:
        m = sm.OLS(Y, X).fit(cov_type="HC1")
    except Exception:
        return np.nan, np.nan, p, k, len(d)
    # 3. Wald: first p x-lags == 0. Column order: const(0), y_l1..y_lk (1..k),
    #    x_l1..x_lk (k+1 .. 2k). Restrict only the first p x-columns.
    xstart = 1 + k
    R = np.zeros((p, X.shape[1]))
    for j in range(p):
        R[j, xstart + j] = 1.0
    try:
        wt = m.wald_test(R, use_f=False)
        wald = float(np.squeeze(wt.statistic))
        ty_p = float(wt.pvalue)
    except Exception:
        return np.nan, np.nan, p, k, len(d)
    return wald, ty_p, p, k, len(d)


def run_toda_yamamoto(voldf, wsent, stat_df):
    # naive F p (from c8f granger_minp) for comparison
    rows = []
    for a in ASSETS:
        for sc in SENT_COLS:
            y = voldf[a]; x = wsent[sc]
            f_naive, p_naive, lag_naive, n = c8f.granger_minp(y, x, WEEKLY_MAXLAG)
            wald, ty_p, p_order, klag, nty = toda_yamamoto(y, x, WEEKLY_MAXLAG, d_max=1)
            rows.append({
                "asset": a, "sentiment": sc,
                "naive_f": f_naive, "naive_p": p_naive, "naive_lag": lag_naive,
                "ty_wald_chi2": wald, "ty_p": ty_p, "ty_var_order_p": p_order,
                "ty_aug_lags_k": klag, "n_weeks": nty,
                "naive_sig_005": (p_naive < ALPHA) if pd.notna(p_naive) else False,
                "ty_sig_005": (ty_p < ALPHA) if pd.notna(ty_p) else False,
            })
    df = pd.DataFrame(rows)
    # BH-FDR on TY p across the 18-pair family
    ty_p = df["ty_p"].values
    valid = pd.notna(ty_p)
    df["ty_fdr_q005"] = False
    df["ty_fdr_q010"] = False
    if valid.sum() > 0:
        rej05, _, *_ = multipletests(ty_p[valid], alpha=0.05, method="fdr_bh")
        rej10, _, *_ = multipletests(ty_p[valid], alpha=0.10, method="fdr_bh")
        df.loc[valid, "ty_fdr_q005"] = rej05
        df.loc[valid, "ty_fdr_q010"] = rej10
    df.to_csv(OUT_DIR / "c12-granger-rigour-toda-yamamoto.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# #D3 Bitcoin / market common-driver control.
# Approach: VAR(p) on [y=vol_asset, x=sentiment] augmented with EXOGENOUS lags
# of a co-movement control (BTC weekly vol for altcoins; market-ex-asset vol for
# BTC/ETH). Test Granger x->y conditional on the exogenous control via a
# restricted-vs-unrestricted SSR F-test on the y-equation regression.
# ----------------------------------------------------------------------------
def conditional_granger(y, x, z, maxlag=WEEKLY_MAXLAG):
    """SSR-F test of x -> y at each lag p, controlling for lags of z (exogenous).

    Unrestricted: y_t ~ const + sum L{1..p} y + sum L{1..p} x + sum L{1..p} z
    Restricted:   drop the x lags. F on the x block. Return min-p across p=1..maxlag.
    """
    df = pd.concat([y, x, z], axis=1).dropna()
    if len(df) < 50:
        return np.nan, np.nan, np.nan, len(df)
    df.columns = ["y", "x", "z"]
    best_p, best_f, best_lag = np.nan, np.nan, np.nan
    for p in range(1, maxlag + 1):
        d = df.copy()
        cols_y = [f"y_l{j}" for j in range(1, p + 1)]
        cols_x = [f"x_l{j}" for j in range(1, p + 1)]
        cols_z = [f"z_l{j}" for j in range(1, p + 1)]
        for j in range(1, p + 1):
            d[f"y_l{j}"] = d["y"].shift(j)
            d[f"x_l{j}"] = d["x"].shift(j)
            d[f"z_l{j}"] = d["z"].shift(j)
        d = d.dropna()
        if len(d) < (3 * p + 5):
            continue
        Y = d["y"].values
        X_un = sm.add_constant(d[cols_y + cols_x + cols_z].values)
        X_re = sm.add_constant(d[cols_y + cols_z].values)
        try:
            m_un = sm.OLS(Y, X_un).fit()
            m_re = sm.OLS(Y, X_re).fit()
        except Exception:
            continue
        ssr_un, ssr_re = m_un.ssr, m_re.ssr
        q = p  # restrictions = number of x lags
        df_den = len(d) - X_un.shape[1]
        if df_den <= 0:
            continue
        F = ((ssr_re - ssr_un) / q) / (ssr_un / df_den)
        pval = float(sps.f.sf(F, q, df_den))
        if np.isnan(best_p) or pval < best_p:
            best_p, best_f, best_lag = pval, F, p
    return best_f, best_p, best_lag, len(df)


def run_btc_control(voldf, wsent):
    rows = []
    for a in ASSETS:
        if a in ("btc", "eth"):
            others = [o for o in ASSETS if o != a]
            z = voldf[others].mean(axis=1)
            ctrl_label = "market_ex_asset_vol"
        else:
            z = voldf["btc"]
            ctrl_label = "btc_vol"
        for sc in SENT_COLS:
            y = voldf[a]; x = wsent[sc]
            # baseline (no control) min-p for reference
            f0, p0, lag0, n0 = c8f.granger_minp(y, x, WEEKLY_MAXLAG)
            fc, pc, lagc, nc = conditional_granger(y, x, z, WEEKLY_MAXLAG)
            rows.append({
                "asset": a, "sentiment": sc, "control": ctrl_label,
                "p_uncontrolled": p0, "lag_uncontrolled": lag0,
                "p_controlled": pc, "lag_controlled": lagc, "n_weeks": nc,
                "uncontrolled_sig_005": (p0 < ALPHA) if pd.notna(p0) else False,
                "controlled_sig_005": (pc < ALPHA) if pd.notna(pc) else False,
            })
    df = pd.DataFrame(rows)
    pc = df["p_controlled"].values
    valid = pd.notna(pc)
    df["controlled_fdr_q005"] = False
    if valid.sum() > 0:
        rej05, _, *_ = multipletests(pc[valid], alpha=0.05, method="fdr_bh")
        df.loc[valid, "controlled_fdr_q005"] = rej05
    df.to_csv(OUT_DIR / "c12-granger-rigour-btc-control.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# #C4 Missingness / initialization-warm-up independence test.
# The source normalized series is unavailable for its first 25 weeks because of
# the 26-observation minimum used by the rolling z-score. The first-lag design
# drops the first of those weeks, leaving 24 warm-up observations for assets
# with full histories and none for later-starting BNB. Test whether membership
# in that warm-up depends on lagged weekly volatility (logit + chi-square).
# ----------------------------------------------------------------------------
def build_imputed_indicator(wsent, week_index):
    """Return 1 only for the normalized sentiment series' leading warm-up.

    ``load_weekly_sentiment`` applies the analysis' pre-cut zero convention, so
    its in-memory values no longer distinguish the rolling-normalization warm-up
    from observed zeros. Read the preserved normalized source column instead.
    In particular, do not infer imputation from an exact zero in a decomposed
    thematic series: a valid normalized value multiplied by a zero theme share
    is an observed zero, not an imputed observation.
    """
    if not wsent.index.equals(week_index):
        raise ValueError("Weekly sentiment and analysis indices are not aligned")

    source = pd.read_csv(
        c2.DATA_DIR / "gdelt.csv",
        usecols=["week_start", "S_gdelt_normalized"],
    )
    source["week_start"] = (
        pd.to_datetime(source["week_start"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    source = source.set_index("week_start").sort_index()
    if source.index.has_duplicates:
        raise ValueError("GDELT normalized-series index contains duplicate weeks")

    warmup = source["S_gdelt_normalized"].isna()
    first_observed = source["S_gdelt_normalized"].first_valid_index()
    if first_observed is None:
        raise ValueError("GDELT normalized series has no observed values")
    if warmup.loc[first_observed:].any():
        raise ValueError("GDELT normalized series has missing values after warm-up")

    aligned = warmup.reindex(week_index)
    if aligned.isna().any():
        raise ValueError(
            "GDELT normalized-series warm-up is not aligned to analysis weeks"
        )
    imp = aligned.astype(int)
    imp.name = "imputed"
    return imp


def run_missingness(voldf, wsent, week_index):
    imp = build_imputed_indicator(wsent, week_index)
    rows = []
    for a in ASSETS:
        vol = voldf[a]
        d = pd.concat([imp, vol.rename("vol")], axis=1).dropna()
        d["vol_l1"] = d["vol"].shift(1)
        d = d.dropna()
        if d["imputed"].nunique() < 2:
            rows.append({"asset": a, "n": len(d),
                         "n_imputed": int(d["imputed"].sum()),
                         "logit_coef_voll1": np.nan, "logit_p": np.nan,
                         "chi2_p": np.nan,
                         "missing_depends_on_vol_005": False})
            continue
        # logit imputed ~ lagged vol
        X = sm.add_constant(d[["vol_l1"]].values)
        try:
            m = sm.Logit(d["imputed"].values, X).fit(disp=0)
            coef = float(m.params[1]); pval = float(m.pvalues[1])
        except Exception:
            coef, pval = np.nan, np.nan
        # chi-square: imputed vs high/low lagged vol (median split)
        med = d["vol_l1"].median()
        hi = (d["vol_l1"] > med).astype(int)
        ct = pd.crosstab(d["imputed"], hi)
        try:
            chi2, chi2p, _, _ = sps.chi2_contingency(ct)
        except Exception:
            chi2p = np.nan
        rows.append({"asset": a, "n": len(d), "n_imputed": int(d["imputed"].sum()),
                     "logit_coef_voll1": coef, "logit_p": pval,
                     "chi2_p": chi2p,
                     "missing_depends_on_vol_005": (pd.notna(pval) and pval < ALPHA)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c12-granger-rigour-missingness.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# #E1 Litigation confound. For XRP/ADA/BNB:
#  (a) PRE-litigation subperiod Granger (native weekly, drop pre-cut zeros).
#  (b) full-sample conditional Granger with litigation-active exogenous indicator.
# ----------------------------------------------------------------------------
def run_litigation(voldf, wsent, week_index):
    rows = []
    survivors = ["xrp", "ada", "bnb"]
    for a in survivors:
        onset = LITIGATION_ONSET[a]
        for sc in SENT_COLS:
            y_full = voldf[a]; x_full = wsent[sc]
            # full-sample baseline
            f_full, p_full, lag_full, n_full = c8f.granger_minp(y_full, x_full, WEEKLY_MAXLAG)

            # (a) PRE-litigation subperiod: weeks strictly before onset, AND after
            # the sentiment warmup (drop the zero-imputed pre-cut weeks so the
            # pre-window isn't dominated by manufactured zeros).
            real_start = pd.Timestamp("2019-06-24")  # first non-imputed week
            mask_pre = (week_index >= real_start) & (week_index < onset)
            yp = y_full[mask_pre]; xp = x_full[mask_pre]
            n_pre = pd.concat([yp, xp], axis=1).dropna().shape[0]
            if n_pre >= 30:
                f_pre, p_pre, lag_pre, _ = c8f.granger_minp(yp, xp, min(WEEKLY_MAXLAG, max(2, n_pre // 6)))
            else:
                f_pre, p_pre, lag_pre = np.nan, np.nan, np.nan

            # (b) litigation-active exogenous indicator in conditional Granger
            end = LITIGATION_END[a]
            litig = pd.Series(
                ((week_index >= onset) & (week_index <= end)).astype(float),
                index=week_index, name="litig")
            f_ctrl, p_ctrl, lag_ctrl, n_ctrl = conditional_granger(
                y_full, x_full, litig, WEEKLY_MAXLAG)

            # (c) DURING-litigation subperiod: weeks from onset through the
            # asset's litigation-active end (sample end for BNB/ADA), same
            # adaptive lag rule as (a). Backs the paper's "significant during
            # it" clause for the XRP split (Sec. 7.4).
            mask_dur = (week_index >= onset) & (week_index <= end)
            yd = y_full[mask_dur]; xd = x_full[mask_dur]
            n_dur = pd.concat([yd, xd], axis=1).dropna().shape[0]
            if n_dur >= 30:
                f_dur, p_dur, lag_dur, _ = c8f.granger_minp(
                    yd, xd, min(WEEKLY_MAXLAG, max(2, n_dur // 6)))
            else:
                f_dur, p_dur, lag_dur = np.nan, np.nan, np.nan

            # (d) PRE-litigation REVERSE direction (vol -> sent), same window
            # and lag rule as (a). Backs the paper's "no detected Granger
            # structure in either direction" clause for the XRP pre-suit split.
            if n_pre >= 30:
                f_prer, p_prer, lag_prer, _ = c8f.granger_minp(
                    xp, yp, min(WEEKLY_MAXLAG, max(2, n_pre // 6)))
            else:
                f_prer, p_prer, lag_prer = np.nan, np.nan, np.nan

            rows.append({
                "asset": a, "sentiment": sc,
                "p_full_sample": p_full, "lag_full": lag_full, "n_full": n_full,
                "n_pre_litig_weeks": n_pre,
                "p_pre_litigation": p_pre, "lag_pre": lag_pre,
                "p_litig_controlled": p_ctrl, "lag_litig_ctrl": lag_ctrl,
                "full_sig_005": (p_full < ALPHA) if pd.notna(p_full) else False,
                "pre_sig_005": (p_pre < ALPHA) if pd.notna(p_pre) else False,
                "litigctrl_sig_005": (p_ctrl < ALPHA) if pd.notna(p_ctrl) else False,
                "n_during_litig_weeks": n_dur,
                "p_during_litigation": p_dur, "lag_during": lag_dur,
                "during_sig_005": (p_dur < ALPHA) if pd.notna(p_dur) else False,
                "p_pre_litigation_reverse": p_prer, "lag_pre_reverse": lag_prer,
                "pre_reverse_sig_005": (p_prer < ALPHA) if pd.notna(p_prer) else False,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c12-granger-rigour-litigation.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# Bottom line
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("C12 GRANGER RIGOUR")
    print("=" * 78)
    voldf, wsent, week_index = build_weekly_panel()

    print("\n[#D2a] Stationarity (ADF + KPSS)")
    stat_df = run_stationarity(voldf, wsent)
    for _, r in stat_df.iterrows():
        print(f"  {r['series']:22s} ADF p={r['adf_p']:.3f} KPSS p={r['kpss_p']:.3f} -> {r['verdict']}")

    print("\n[#D2b] Toda-Yamamoto lag-augmented VAR (sent -> vol)")
    ty_df = run_toda_yamamoto(voldf, wsent, stat_df)
    for _, r in ty_df.iterrows():
        flag = "SIG" if r["ty_sig_005"] else "   "
        print(f"  {r['asset']} {r['sentiment']:20s} naive p={r['naive_p']:.4f} -> "
              f"TY p={r['ty_p']:.4f} (VAR p={r['ty_var_order_p']}, k={r['ty_aug_lags_k']}) [{flag}]")

    print("\n[#D3] BTC / market common-driver control (sent -> vol | lagged co-mvmt)")
    btc_df = run_btc_control(voldf, wsent)
    for _, r in btc_df.iterrows():
        flag = "SIG" if r["controlled_sig_005"] else "   "
        print(f"  {r['asset']} {r['sentiment']:20s} uncond p={r['p_uncontrolled']:.4f} -> "
              f"cond({r['control']}) p={r['p_controlled']:.4f} [{flag}]")

    print("\n[#C4] Missingness: imputed-indicator ~ lagged vol")
    miss_df = run_missingness(voldf, wsent, week_index)
    for _, r in miss_df.iterrows():
        print(f"  {r['asset']} n_imp={r.get('n_imputed','?')} logit(vol_l1) coef={r['logit_coef_voll1']:.3f} "
              f"p={r['logit_p']:.4f} chi2 p={r['chi2_p']:.4f}")

    print("\n[#E1] Litigation confound (XRP/ADA/BNB)")
    litig_df = run_litigation(voldf, wsent, week_index)
    for _, r in litig_df.iterrows():
        print(f"  {r['asset']} {r['sentiment']:20s} full p={r['p_full_sample']:.4f} | "
              f"pre-litig p={r['p_pre_litigation']} (n={r['n_pre_litig_weeks']}) | "
              f"litig-ctrl p={r['p_litig_controlled']:.4f}")

    # -------- BOTTOM LINE TALLY --------
    # Baseline universe: 18 (asset x sentiment) sent->vol pairs.
    # FDR-7 survivors (from c8g, q<0.05 weekly_sent_to_vol):
    base = pd.read_csv(OUT_DIR / "c8f-weekly-granger.csv")
    p_base = base["p_sent_to_vol"].values
    rej_base, _, *_ = multipletests(p_base, alpha=0.05, method="fdr_bh")
    base["fdr_q005"] = rej_base
    fdr7 = base[base["fdr_q005"]][["asset", "sentiment"]].apply(tuple, axis=1).tolist()

    n_raw_18 = int((p_base < ALPHA).sum())
    n_fdr_7 = int(rej_base.sum())

    # TY: how many of 18 / of the FDR-7 survive
    ty_df["pair"] = list(zip(ty_df["asset"], ty_df["sentiment"]))
    n_ty_18 = int(ty_df["ty_sig_005"].sum())
    n_ty_fdr18 = int(ty_df["ty_fdr_q005"].sum())
    ty_survivors = set(ty_df[ty_df["ty_sig_005"]]["pair"])
    n_ty_of7 = sum(1 for p in fdr7 if p in ty_survivors)

    # BTC control
    btc_df["pair"] = list(zip(btc_df["asset"], btc_df["sentiment"]))
    n_btc_18 = int(btc_df["controlled_sig_005"].sum())
    btc_survivors = set(btc_df[btc_df["controlled_sig_005"]]["pair"])
    n_btc_of7 = sum(1 for p in fdr7 if p in btc_survivors)

    # Joint: survive BOTH TY and BTC-control (raw 0.05)
    joint = ty_survivors & btc_survivors
    n_joint_18 = len(joint)
    n_joint_of7 = sum(1 for p in fdr7 if p in joint)

    # Litigation: of the FDR-7 (all xrp/ada/bnb), how many survive pre-litig &
    # litig-controlled
    litig_df["pair"] = list(zip(litig_df["asset"], litig_df["sentiment"]))
    litig_lookup_pre = dict(zip(litig_df["pair"], litig_df["pre_sig_005"]))
    litig_lookup_ctrl = dict(zip(litig_df["pair"], litig_df["litigctrl_sig_005"]))
    litig_lookup_dur = dict(zip(litig_df["pair"], litig_df["during_sig_005"]))
    n_pre_of7 = sum(1 for p in fdr7 if litig_lookup_pre.get(p, False))
    n_ctrl_of7 = sum(1 for p in fdr7 if litig_lookup_ctrl.get(p, False))
    n_dur_of7 = sum(1 for p in fdr7 if litig_lookup_dur.get(p, False))

    # Full gauntlet of the 7: survive TY AND BTC-control AND litig-controlled
    n_gauntlet_of7 = sum(
        1 for p in fdr7
        if (p in ty_survivors) and (p in btc_survivors) and litig_lookup_ctrl.get(p, False))
    gauntlet_pairs = [p for p in fdr7
                      if (p in ty_survivors) and (p in btc_survivors)
                      and litig_lookup_ctrl.get(p, False)]

    summary = pd.DataFrame([
        {"test": "raw weekly p<0.05 (18 pairs)", "n_of_18": n_raw_18, "n_of_fdr7": n_fdr_7},
        {"test": "BH-FDR q<0.05 (the 7 survivors)", "n_of_18": n_fdr_7, "n_of_fdr7": n_fdr_7},
        {"test": "Toda-Yamamoto p<0.05", "n_of_18": n_ty_18, "n_of_fdr7": n_ty_of7},
        {"test": "Toda-Yamamoto + BH-FDR q<0.05", "n_of_18": n_ty_fdr18, "n_of_fdr7": np.nan},
        {"test": "BTC/market control p<0.05", "n_of_18": n_btc_18, "n_of_fdr7": n_btc_of7},
        {"test": "TY AND BTC-control p<0.05", "n_of_18": n_joint_18, "n_of_fdr7": n_joint_of7},
        {"test": "litigation pre-period p<0.05", "n_of_18": np.nan, "n_of_fdr7": n_pre_of7},
        {"test": "litigation during-period p<0.05", "n_of_18": np.nan, "n_of_fdr7": n_dur_of7},
        {"test": "litigation-controlled p<0.05", "n_of_18": np.nan, "n_of_fdr7": n_ctrl_of7},
        {"test": "FULL GAUNTLET (TY+BTC+litig-ctrl)", "n_of_18": np.nan, "n_of_fdr7": n_gauntlet_of7},
    ])
    summary.to_csv(OUT_DIR / "c12-granger-rigour-bottomline.csv", index=False)

    print("\n" + "=" * 78)
    print("BOTTOM LINE")
    print("=" * 78)
    print(f"FDR-7 survivors (baseline): {fdr7}")
    for _, r in summary.iterrows():
        print(f"  {r['test']:38s} of18={r['n_of_18']}  of-FDR7={r['n_of_fdr7']}")
    print(f"\n  Pairs surviving the FULL gauntlet: {gauntlet_pairs}")
    print("\nSaved: c12-granger-rigour-{stationarity,toda-yamamoto,btc-control,"
          "missingness,litigation,bottomline}.csv")


if __name__ == "__main__":
    main()
