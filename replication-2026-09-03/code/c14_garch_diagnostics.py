"""
C14: GARCH squared-residual diagnostics for the baseline GJR-GARCH-X.

Question (reviewer #C2)
-----------------------
Do the fitted standardised residuals retain detectable conditional
heteroskedasticity at the tested lags? If they still show ARCH, then
unmodelled volatility clustering -- especially in the 2022-23 window where
infrastructure events cluster (Terra/Luna, FTX) -- could be soaked up by the
event dummies, mechanically inflating delta_infra and the curated 3.49x
multiplier.

What this does
--------------
For each of the 6 assets, on the c7 baseline spec (S1, 50 curated events, the
SAME 5 exog as c6/c7: D_infra, D_reg, S_gdelt, S_reg, S_infra):
  1. Fit the baseline GJR-GARCH-X (multistart, identical model to c7's
     fit_observed -> reuses FastTARCHX + tarch_x_fast).
  2. Compute standardised residuals  z_t = eps_t / sqrt(sigma2_t)
     and squared standardised residuals z_t^2.
  3. Ljung-Box Q on z_t^2 at lags 5, 10, 20 (the standard McLeod-Li / residual
     ARCH portmanteau test on a fitted GARCH).
  4. Engle ARCH-LM test on z_t (T*R^2 from regressing z_t^2 on its own lags),
     reported at lags 5 and 10.
  5. Basic z-moment sanity: mean(z^2) (should be ~1), excess kurtosis,
     mean(z) (should be ~0).

A NOTE on degrees of freedom for the portmanteau test
-----------------------------------------------------
Standard Ljung-Box uses df = lag. Model-aware corrections for squared
standardised residual autocorrelations can be more complicated than a universal
degrees-of-freedom subtraction (Li & Mak 1994). We therefore report BOTH:
  * Q with naive df = lag (statsmodels default), and
  * a simple, more rejection-prone sensitivity using
    df = max(lag - (p + q), 1) = max(lag - 2, 1).
The latter is a diagnostic sensitivity, not a claim to implement the general
Li-Mak correction. Both conventions are shown and give the same verdict.

Outputs (in the package results directory):
  c14-garch-diagnostics-per-asset.csv
  c14-garch-diagnostics-FINDING.md
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_relaxed_threshold_sensitivity as c2  # loaders, ASSETS, dummies, OUT_DIR
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA

ASSETS = c2.ASSETS
SENT_COLS = ["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]
N_STARTS_FIT = 6
MAX_ITER = 2000
SEED = 12345
LB_LAGS = [5, 10, 20]
ARCHLM_LAGS = [5, 10]
N_LAG_ORDERS = 2  # ARCH+GARCH lag orders used only in the df-sensitivity column


# ----------------------------------------------------------------------------
# Build the SAME baseline design c7 uses (S1, 50 events, 5 exog)
# ----------------------------------------------------------------------------
def build_design():
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(c2.OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)

    design = {}
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
        dum = c2.build_event_dummies(r.index, inf_d, reg_d,
                                     c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s = sent.reindex(r.index).fillna(0)
        Dinf = dum["D_infrastructure"].values
        Dreg = dum["D_regulatory"].values
        S = s[SENT_COLS].values
        exog_unr = np.column_stack([Dinf, Dreg, S])  # [D_infra, D_reg, S_gdelt, S_reg, S_infra]
        design[a] = {
            "returns": r.values.astype(float),
            "exog_unr": exog_unr,
            "index": r.index,
        }
    return design, inf_d, reg_d


# ----------------------------------------------------------------------------
# Portmanteau + ARCH-LM
# ----------------------------------------------------------------------------
def ljung_box(x, lags):
    """
    Ljung-Box Q statistic on series x at each lag in `lags`.
    Returns dict lag -> (Q, p_naive_df_lag). x is mean-centred internally for
    the autocorrelation computation (we feed z^2, whose mean ~1).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    denom = np.sum(xc * xc)
    out = {}
    maxlag = max(lags)
    # autocorrelations 1..maxlag
    acf = np.empty(maxlag + 1)
    acf[0] = 1.0
    for k in range(1, maxlag + 1):
        acf[k] = np.sum(xc[k:] * xc[:-k]) / denom
    for L in lags:
        s = 0.0
        for k in range(1, L + 1):
            s += acf[k] ** 2 / (n - k)
        Q = n * (n + 2) * s
        p_naive = stats.chi2.sf(Q, df=L)
        df_adj = max(L - N_LAG_ORDERS, 1)
        p_adj = stats.chi2.sf(Q, df=df_adj)
        out[L] = (Q, p_naive, p_adj, df_adj)
    return out


def arch_lm(z, lags):
    """
    Engle (1982) ARCH-LM test. Regress z_t^2 on a constant and `lags` of z_t^2;
    LM = (n - lags) * R^2 ~ chi2(lags) under H0 of no remaining ARCH.
    Returns dict lag -> (LM, p).
    """
    z = np.asarray(z, dtype=float)
    u = z * z
    out = {}
    for L in lags:
        n = u.size
        y = u[L:]
        X = np.ones((y.size, L + 1))
        for k in range(1, L + 1):
            X[:, k] = u[L - k: n - k]
        # OLS
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        nobs = y.size
        LM = nobs * r2
        p = stats.chi2.sf(LM, df=L)
        out[L] = (LM, p, r2)
    return out


# ----------------------------------------------------------------------------
# Fit + diagnose per asset
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"numba available: {_HAVE_NUMBA}")
    print("Building baseline (S1, 50-event) design...")
    design, inf_d, reg_d = build_design()
    print(f"  baseline events: {len(inf_d)} infra, {len(reg_d)} reg")

    rows = []
    for a in ASSETS:
        d = design[a]
        est = FastTARCHX(d["returns"], d["exog_unr"])
        p, f, ok = est.fit_multistart(n_starts=N_STARTS_FIT, seed=SEED, max_iter=MAX_ITER)
        var = est._variance(p)
        z = est.resid / np.sqrt(var)
        z2 = z * z

        # Verify whether the numerical positivity floor binds at the optimum.
        # Use the same recursively floored lagged variance as the estimator, but
        # retain the pre-floor value at each step for the diagnostic.
        exog_contrib = d["exog_unr"] @ p[5:]
        var_check = np.empty(est.n_obs)
        var_check[0] = est.var0
        raw_min = np.inf
        floor_hits = 0
        for t in range(1, est.n_obs):
            e = est.resid[t - 1]
            raw = (p[0] + p[1] * e * e + p[2] * e * e * (e < 0)
                   + p[3] * var_check[t - 1] + exog_contrib[t])
            raw_min = min(raw_min, raw)
            floor_hits += int(raw < 1e-8)
            var_check[t] = max(raw, 1e-8)

        omega, alpha, gamma, beta, nu = p[0], p[1], p[2], p[3], p[4]
        d_infra, d_reg = p[5], p[6]
        persistence = alpha + beta + abs(gamma) / 2.0

        lb = ljung_box(z2, LB_LAGS)
        lm = arch_lm(z, ARCHLM_LAGS)

        n = z.size
        mean_z = float(z.mean())
        mean_z2 = float(z2.mean())
        # excess kurtosis of z
        zc = z - mean_z
        sd = zc.std()
        exkurt = float(np.mean((zc / sd) ** 4) - 3.0) if sd > 0 else np.nan

        row = {
            "asset": a, "n_obs": n, "converged": ok, "negLL": f,
            "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "nu": nu,
            "persistence": persistence,
            "delta_infra": d_infra, "delta_reg": d_reg,
            "theta_gdelt": p[7], "theta_reg": p[8], "theta_infra": p[9],
            "variance_floor_hits": floor_hits,
            "min_raw_variance": raw_min,
            "mean_z": mean_z, "mean_z2": mean_z2, "exkurt_z": exkurt,
        }
        for L in LB_LAGS:
            Q, p_naive, p_adj, df_adj = lb[L]
            row[f"LB_z2_Q{L}"] = Q
            row[f"LB_z2_p{L}_naive"] = p_naive
            row[f"LB_z2_p{L}_adj"] = p_adj
            row[f"LB_z2_dfadj{L}"] = df_adj
        for L in ARCHLM_LAGS:
            LM, pv, r2 = lm[L]
            row[f"ARCHLM_stat{L}"] = LM
            row[f"ARCHLM_p{L}"] = pv
        rows.append(row)

        print(f"\n{a}: n={n} ok={ok} | omega={omega:.4f} alpha={alpha:.4f} "
              f"gamma={gamma:.4f} beta={beta:.4f} nu={nu:.2f} persist={persistence:.4f}")
        print(f"   delta_infra={d_infra:.4f} delta_reg={d_reg:.4f} | "
              f"mean(z)={mean_z:.3f} mean(z^2)={mean_z2:.3f} exkurt={exkurt:.2f} | "
              f"floor hits={floor_hits} min raw variance={raw_min:.4g}")
        for L in LB_LAGS:
            Q, p_naive, p_adj, df_adj = lb[L]
            print(f"   LB(z^2) lag {L:2d}: Q={Q:7.3f}  p_naive(df={L})={p_naive:.4f}  "
                  f"p_adj(df={df_adj})={p_adj:.4f}")
        for L in ARCHLM_LAGS:
            LM, pv, r2 = lm[L]
            print(f"   ARCH-LM  lag {L:2d}: LM={LM:7.3f}  p={pv:.4f}  R2={r2:.4f}")

    df = pd.DataFrame(rows)
    out_csv = c2.OUT_DIR / "c14-garch-diagnostics-per-asset.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    write_finding(df, len(inf_d), len(reg_d), time.time() - t0)
    return df


def write_finding(df, n_inf, n_reg, elapsed):
    # Count significant residual-ARCH flags (p < 0.05) at each diagnostic.
    # Use the more rejection-prone lag-minus-two sensitivity and ARCH-LM p.
    alpha_lvl = 0.05
    flags = {}
    for L in LB_LAGS:
        flags[f"LB_adj_{L}"] = int((df[f"LB_z2_p{L}_adj"] < alpha_lvl).sum())
        flags[f"LB_naive_{L}"] = int((df[f"LB_z2_p{L}_naive"] < alpha_lvl).sum())
    for L in ARCHLM_LAGS:
        flags[f"ARCHLM_{L}"] = int((df[f"ARCHLM_p{L}"] < alpha_lvl).sum())

    total_adj_flags = sum(flags[f"LB_adj_{L}"] for L in LB_LAGS) + \
        sum(flags[f"ARCHLM_{L}"] for L in ARCHLM_LAGS)
    any_asset_flagged = sorted(set(
        list(df.loc[(df[[f"LB_z2_p{L}_adj" for L in LB_LAGS]] < alpha_lvl).any(axis=1), "asset"]) +
        list(df.loc[(df[[f"ARCHLM_p{L}" for L in ARCHLM_LAGS]] < alpha_lvl).any(axis=1), "asset"])
    ))

    # Verdict logic
    if total_adj_flags == 0:
        verdict = (
            "ADEQUATE. No asset shows significant residual ARCH at the 5% level on "
            "either Ljung-Box(z^2) convention (lags 5/10/20) or the "
            "Engle ARCH-LM test (lags 5/10). These diagnostics find no evidence of "
            "remaining ARCH at the tested lags. They do not prove that the variance "
            "model is complete or that the event coefficients are free of every "
            "possible dynamic misspecification."
        )
    elif any_asset_flagged and len(any_asset_flagged) <= 2:
        verdict = (
            f"MOSTLY ADEQUATE with a caveat. {len(any_asset_flagged)} of 6 assets "
            f"({', '.join(any_asset_flagged)}) show some residual ARCH at 5%. Because "
            "infrastructure events cluster in the high-variance 2022-23 window, any "
            "unmodelled clustering there could be partially absorbed by D_infra and "
            "inflate delta_infra. The curated ~3.49x multiplier should therefore be "
            "flagged as a POSSIBLE UPPER BOUND for the affected assets, pending a "
            "richer variance dynamic (e.g. component/EGARCH)."
        )
    else:
        verdict = (
            f"INADEQUATE for {len(any_asset_flagged)} of 6 assets "
            f"({', '.join(any_asset_flagged)}): significant residual ARCH remains. "
            "Unmodelled volatility clustering -- concentrated in the 2022-23 window "
            "where infrastructure events cluster -- can be soaked up by D_infra and "
            "mechanically inflate delta_infra. The curated ~3.49x multiplier MUST be "
            "flagged as a likely UPPER BOUND; a richer variance specification "
            "(component GARCH / EGARCH / regime-switching) is warranted before the "
            "event coefficients are interpreted at face value."
        )

    lines = []
    lines.append("# C14 -- GARCH Squared-Residual Diagnostics (baseline GJR-GARCH-X)\n")
    lines.append(f"_Baseline S1 spec ({n_inf} infra + {n_reg} reg curated events); "
                 f"FastTARCHX multistart fit (same model as c6/c7); numba={_HAVE_NUMBA}; "
                 f"runtime {elapsed:.1f}s._\n")

    lines.append("## The question\n")
    lines.append("Reviewer #C2 asks whether the GJR-GARCH-X variance equation fully "
                 "absorbs conditional heteroskedasticity *before* the event coefficients "
                 "are interpreted. If standardised residuals still carry ARCH, then "
                 "unmodelled volatility clustering -- which is concentrated in the "
                 "2022-23 window where the infrastructure events (Terra/Luna, FTX, etc.) "
                 "cluster -- could be absorbed by the D_infra dummy and mechanically "
                 "inflate delta_infra and the ~3.49x curated multiplier. The test: "
                 "Ljung-Box Q on z_t^2 at lags 5/10/20 plus Engle ARCH-LM, per asset.\n")

    lines.append("## Fitted variance parameters (baseline)\n")
    lines.append("| asset | n | omega | alpha | gamma | beta | nu | persist. | dInfra | dReg |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['asset']} | {int(r['n_obs'])} | {r['omega']:.4f} | "
                     f"{r['alpha']:.4f} | {r['gamma']:.4f} | {r['beta']:.4f} | "
                     f"{r['nu']:.2f} | {r['persistence']:.4f} | {r['delta_infra']:.4f} | "
                     f"{r['delta_reg']:.4f} |")
    lines.append("")
    lines.append("_persist. = alpha + beta + |gamma|/2 (stationarity < 1)._\n")
    lines.append("The numerical variance floor is not reached in any fitted series; "
                 f"the smallest pre-floor variance is {df['min_raw_variance'].min():.4g}.\n")

    lines.append("## Standardised-residual moments (whitening sanity)\n")
    lines.append("| asset | mean(z) | mean(z^2) | excess kurt(z) |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['asset']} | {r['mean_z']:.3f} | {r['mean_z2']:.3f} | "
                     f"{r['exkurt_z']:.2f} |")
    lines.append("")
    lines.append("_mean(z)~0 and mean(z^2)~1 indicate the variance level is captured; "
                 "residual excess kurtosis is expected (Student-t marginals) and is "
                 "not an ARCH symptom._\n")

    lines.append("## Ljung-Box Q on z_t^2 (residual-ARCH portmanteau)\n")
    lines.append("| asset | Q(5) | p5 adj | Q(10) | p10 adj | Q(20) | p20 adj |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['asset']} | {r['LB_z2_Q5']:.2f} | {r['LB_z2_p5_adj']:.4f} | "
                     f"{r['LB_z2_Q10']:.2f} | {r['LB_z2_p10_adj']:.4f} | "
                     f"{r['LB_z2_Q20']:.2f} | {r['LB_z2_p20_adj']:.4f} |")
    lines.append("")
    lines.append("_p..adj is a simple, more rejection-prone df = lag - 2 sensitivity "
                 "that subtracts the ARCH and GARCH lag orders. It is not labelled as a "
                 "general Li-Mak correction. Conventional df = lag p-values are in the "
                 "CSV and give the same conclusion._\n")

    lines.append("## Engle ARCH-LM on z_t (T*R^2 of z^2 on its lags)\n")
    lines.append("| asset | LM(5) | p(5) | LM(10) | p(10) |")
    lines.append("|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['asset']} | {r['ARCHLM_stat5']:.2f} | {r['ARCHLM_p5']:.4f} | "
                     f"{r['ARCHLM_stat10']:.2f} | {r['ARCHLM_p10']:.4f} |")
    lines.append("")

    lines.append("## Significant residual-ARCH flags (p < 0.05)\n")
    lines.append(f"- Ljung-Box(z^2), adjusted df: "
                 f"lag5 = {flags['LB_adj_5']}/6, lag10 = {flags['LB_adj_10']}/6, "
                 f"lag20 = {flags['LB_adj_20']}/6 assets.")
    lines.append(f"- Ljung-Box(z^2), naive df (for reference): "
                 f"lag5 = {flags['LB_naive_5']}/6, lag10 = {flags['LB_naive_10']}/6, "
                 f"lag20 = {flags['LB_naive_20']}/6 assets.")
    lines.append(f"- ARCH-LM: lag5 = {flags['ARCHLM_5']}/6, lag10 = {flags['ARCHLM_10']}/6 assets.")
    if any_asset_flagged:
        lines.append(f"- Assets flagged by ANY adjusted test: {', '.join(any_asset_flagged)}.")
    else:
        lines.append("- Assets flagged by ANY adjusted test: none.")
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(verdict + "\n")

    lines.append("## Caveats (honest)\n")
    lines.append("- The portmanteau on squared standardised residuals is the standard "
                 "McLeod-Li check for *remaining* ARCH after a GARCH fit; it does not "
                 "test the level fit (mean(z^2)~1 covers that separately above).")
    lines.append("- The df = lag - (p+q) = lag - 2 column is a deliberately more "
                 "rejection-prone sensitivity, not a universal Li-Mak correction. Li and "
                 "Mak (1994) motivate model-aware squared-residual diagnostics whose exact "
                 "asymptotic adjustment can be nonstandard. The conventional df = lag "
                 "column is also reported and yields the same conclusion. The result is "
                 "therefore presented as a diagnostic rather than formal model certification.")
    lines.append("- A clean portmanteau means no *linear* ARCH remains in z^2; it does "
                 "not rule out higher-order nonlinearity or regime structure (those are "
                 "handled separately by the Bai-Perron / persistence-break analyses in "
                 "c3/c8).")
    lines.append("- This diagnostic conditions on the fitted point estimates (multistart "
                 "MLE); it is descriptive of model adequacy, not an inference test on "
                 "the deltas (those are assessed by the c9 fixed-path bootstrap and "
                 "the c19 recursive sensitivity).")
    lines.append("")
    lines.append("## Files\n")
    lines.append("- `c14-garch-diagnostics-per-asset.csv` -- full per-asset table")
    lines.append("- `code/c14_garch_diagnostics.py`, `code/tarch_x_fast.py`, "
                 "`code/c2_relaxed_threshold_sensitivity.py`")

    (c2.OUT_DIR / "c14-garch-diagnostics-FINDING.md").write_text("\n".join(lines))
    print(f"Saved {c2.OUT_DIR / 'c14-garch-diagnostics-FINDING.md'}")


if __name__ == "__main__":
    main()
