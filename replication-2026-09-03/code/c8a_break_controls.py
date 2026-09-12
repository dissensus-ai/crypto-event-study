"""
C8a: Break-regime controls in the PRIMARY spec (the headline-mover).
====================================================================

The reviewer's worry: the infrastructure-vs-regulatory asymmetry could be an
artefact of crisis-period baseline variance. Infrastructure events (exchange
hacks, the FTX collapse, the Terra/Luna implosion) cluster in the high-variance
2021-2022 window. If the variance equation has no way to absorb that regime-level
baseline, the D_infrastructure dummy will soak it up and look large.

This script re-fits the GJR-GARCH-X per asset on the baseline S1 50-event curated
sample, but adds regime-indicator dummies (from the c3 Bai-Perron / PELT
cond_variance break dates) to the VARIANCE equation alongside
[D_infrastructure, D_regulatory, S_gdelt, S_reg, S_infra].

Two control variants:
  (A) FULL regime dummies: one indicator per detected cond_variance regime
      segment, first segment dropped as baseline.
  (B) SINGLE high-variance-segment dummy (legacy machine key ``crisis``): a
      0/1 flag from each asset's last 2021 conditional-variance break to its
      first 2022-or-later break. Four windows end around FTX; BNB and ADA end
      in July 2022, so this is not a pure FTX dummy.

KEY QUESTION: once this asset-specific high-variance baseline is absorbed, how
much does the infra coefficient drop? Does the multiplier survive vs the 3.49x
uncontrolled baseline?

Uses the canonical TARCHXEstimator (with model SEs) so we can report per-asset
deltas and significance, exactly like c2/c6.

Outputs:
    results/c8a-break-controls-per-asset.csv
    results/c8a-break-controls-summary.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2  # noqa: E402

OUT_DIR = c2.OUT_DIR
ASSETS = c2.ASSETS


def load_regime_dummies(date_index, asset, c3_df, variant="full"):
    """
    Build regime-indicator dummies for one asset from c3 cond_variance breaks.

    variant="full": one dummy per regime segment after the first (first = baseline).
    variant="crisis": legacy machine key for one asset-specific high-variance
        segment, defined from the last 2021 break to the first 2022-or-later
        break. Four asset windows end around FTX; BNB and ADA end in July 2022.
        Falls back to the segment immediately preceding the first 2022-or-later
        break if no clean 2021 start exists.
    """
    idx = pd.DatetimeIndex(date_index)
    breaks = (c3_df[(c3_df["asset"] == asset) & (c3_df["series_type"] == "cond_variance")]
              ["break_date"].sort_values().tolist())
    breaks = [pd.Timestamp(b) for b in breaks]

    if variant == "full":
        # segments: (-inf, b0), [b0, b1), ..., [b_{k-1}, +inf)
        edges = [idx.min()] + breaks + [idx.max() + pd.Timedelta(days=1)]
        cols = {}
        for i in range(1, len(edges) - 1):  # drop segment 0 as baseline
            lo, hi = edges[i], edges[i + 1]
            d = pd.Series(0.0, index=idx, name=f"regime_{asset}_{i}")
            d.loc[(idx >= lo) & (idx < hi)] = 1.0
            cols[d.name] = d
        return pd.DataFrame(cols, index=idx)

    elif variant == "crisis":
        # Asset-specific high-variance window: use the LAST 2021 break as start
        # and the FIRST 2022-or-later break as end. This ends around FTX for four
        # assets, but in July 2022 for BNB and ADA.
        starts_2021 = [b for b in breaks if b.year == 2021]
        ends_2022p = [b for b in breaks if b.year >= 2022]
        if starts_2021 and ends_2022p:
            lo = max(starts_2021)
            hi = min(ends_2022p)
        elif ends_2022p:
            # bnb/ada have 2020/2021/2022 mid-year breaks; bracket the segment
            # ending at the first >=2022 break, starting at the prior break.
            hi = min(ends_2022p)
            prior = [b for b in breaks if b < hi]
            lo = max(prior) if prior else idx.min()
        else:
            lo, hi = idx.min(), idx.max() + pd.Timedelta(days=1)
        d = pd.Series(0.0, index=idx, name=f"crisis_{asset}")
        d.loc[(idx >= lo) & (idx < hi)] = 1.0
        return pd.DataFrame({d.name: d}, index=idx), (lo, hi)
    else:
        raise ValueError(variant)


def fit_asset(returns, exog):
    est = c2.TARCHXEstimator(returns, exog)
    res = est.estimate(method="SLSQP", max_iter=2000)
    return res


def summarize(label, di, dr, p_i=None, p_r=None):
    infra = np.array(di, float)
    reg = np.array(dr, float)
    inf_mean, reg_mean = np.nanmean(infra), np.nanmean(reg)
    mult = inf_mean / reg_mean if reg_mean != 0 else np.nan
    t, p = stats.ttest_ind(infra, reg, equal_var=False, nan_policy="omit")
    return {
        "variant": label,
        "mean_infra": inf_mean, "mean_reg": reg_mean,
        "median_infra": np.nanmedian(infra), "median_reg": np.nanmedian(reg),
        "multiplier": mult, "welch_t": t, "welch_p": p,
    }


def main():
    print("Loading panel/sentiment/events...")
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    c3 = pd.read_csv(OUT_DIR / "c3-bai-perron-results.csv"); c3["break_date"] = pd.to_datetime(c3["break_date"])

    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)
    print(f"baseline: {len(inf_d)} infra, {len(reg_d)} reg events")

    per_asset = []
    # accumulate per-variant deltas
    acc = {"baseline": ([], [], [], []),
           "full_regime": ([], [], [], []),
           "crisis_dummy": ([], [], [], [])}
    crisis_windows = {}

    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
        dum = c2.build_event_dummies(r.index, inf_d, reg_d,
                                     c2.WINDOW_DAYS_BEFORE, c2.WINDOW_DAYS_AFTER)
        s = sent.reindex(r.index).fillna(0)
        base_exog = pd.concat(
            [dum[["D_infrastructure", "D_regulatory"]],
             s[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
            axis=1).fillna(0)

        # ---- (0) baseline (no regime controls) ----
        res0 = fit_asset(r, base_exog)
        di0 = res0.params.get("D_infrastructure", np.nan)
        dr0 = res0.params.get("D_regulatory", np.nan)
        pi0 = res0.pvalues.get("D_infrastructure", np.nan)
        pr0 = res0.pvalues.get("D_regulatory", np.nan)
        acc["baseline"][0].append(di0); acc["baseline"][1].append(dr0)
        acc["baseline"][2].append(pi0); acc["baseline"][3].append(pr0)

        # ---- (A) full regime dummies ----
        reg_full = load_regime_dummies(r.index, a, c3, variant="full")
        exogA = pd.concat([base_exog, reg_full], axis=1).fillna(0)
        resA = fit_asset(r, exogA)
        diA = resA.params.get("D_infrastructure", np.nan)
        drA = resA.params.get("D_regulatory", np.nan)
        piA = resA.pvalues.get("D_infrastructure", np.nan)
        prA = resA.pvalues.get("D_regulatory", np.nan)
        acc["full_regime"][0].append(diA); acc["full_regime"][1].append(drA)
        acc["full_regime"][2].append(piA); acc["full_regime"][3].append(prA)

        # ---- (B) single high-variance dummy (legacy key: crisis) ----
        crisis_df, (lo, hi) = load_regime_dummies(r.index, a, c3, variant="crisis")
        crisis_windows[a] = (lo, hi)
        exogB = pd.concat([base_exog, crisis_df], axis=1).fillna(0)
        resB = fit_asset(r, exogB)
        diB = resB.params.get("D_infrastructure", np.nan)
        drB = resB.params.get("D_regulatory", np.nan)
        piB = resB.pvalues.get("D_infrastructure", np.nan)
        prB = resB.pvalues.get("D_regulatory", np.nan)
        acc["crisis_dummy"][0].append(diB); acc["crisis_dummy"][1].append(drB)
        acc["crisis_dummy"][2].append(piB); acc["crisis_dummy"][3].append(prB)

        n_regimes = reg_full.shape[1] + 1
        print(f"  {a}: base dInfra={di0:.3f}(p{pi0:.3f}) dReg={dr0:.3f}(p{pr0:.3f}) | "
              f"fullReg({n_regimes}seg) dInfra={diA:.3f} dReg={drA:.3f} | "
              f"highVariance[{lo.date()}..{hi.date()}] "
              f"dInfra={diB:.3f} dReg={drB:.3f} (legacy key: crisis)")

        per_asset.append({
            "asset": a, "n_regimes": n_regimes,
            "crisis_lo": lo.date(), "crisis_hi": hi.date(),
            "base_dInfra": di0, "base_dReg": dr0, "base_pInfra": pi0, "base_pReg": pr0,
            "base_omega": res0.params.get("omega", np.nan),
            "fullReg_dInfra": diA, "fullReg_dReg": drA, "fullReg_pInfra": piA, "fullReg_pReg": prA,
            "fullReg_omega": resA.params.get("omega", np.nan),
            "crisis_dInfra": diB, "crisis_dReg": drB, "crisis_pInfra": piB, "crisis_pReg": prB,
            "crisis_coef": resB.params.get(f"crisis_{a}", np.nan),
            "crisis_omega": resB.params.get("omega", np.nan),
        })

    df_pa = pd.DataFrame(per_asset)
    df_pa.to_csv(OUT_DIR / "c8a-break-controls-per-asset.csv", index=False)

    summary = [
        summarize("baseline_no_controls", *acc["baseline"][:2]),
        summarize("full_regime_dummies", *acc["full_regime"][:2]),
        summarize("single_crisis_dummy", *acc["crisis_dummy"][:2]),
    ]
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUT_DIR / "c8a-break-controls-summary.csv", index=False)

    print("\n=== C8a SUMMARY (cross-asset) ===")
    for row in summary:
        print(f"  {row['variant']:24s} infra={row['mean_infra']:.3f} reg={row['mean_reg']:.3f} "
              f"mult={row['multiplier']:.2f}x  welch_p={row['welch_p']:.4f}")
    print(f"\nUncontrolled baseline reference: ~3.49x")
    print(f"Saved: c8a-break-controls-per-asset.csv, c8a-break-controls-summary.csv")


if __name__ == "__main__":
    main()
