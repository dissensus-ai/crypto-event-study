"""
C8b: Anticipation confound (asymmetric event windows).
======================================================

Reviewer's worry: regulatory events are frequently leaked / anticipated (a
hearing is scheduled, a draft bill circulates, an SEC suit is trailed in the
press). A symmetric [-3, +3] window then UNDERCOUNTS the pre-event regulatory
volatility -- it falls outside the window -- so D_regulatory looks small and the
infra/reg multiplier looks large for a mechanical, not economic, reason.

Test: keep infrastructure events on the symmetric [-3, +3] window, but give
REGULATORY events a progressively longer PRE-window. Sweep the reg pre-window
over {3, 5, 7, 10} trading days (reg post-window stays +3). Refit GJR-GARCH-X
per asset on the baseline S1 50-event sample at each setting and report the
cross-asset multiplier.

If the multiplier falls toward 1 as the reg pre-window lengthens, anticipation
is a real confound. If it is roughly flat, anticipation does not explain the gap.

NOTE on a "surprise" subset: the event metadata (events.csv) and the candidate
census (c1-dropout-census.csv) carry NO anticipated/surprise flag -- only
event_id/date/label/title/type. So a surprise-only multiplier cannot be computed
from existing data without hand-coding each event's anticipation status, which is
out of scope for this control battery. Reported as a data limitation.

Uses FastTARCHX for the point deltas (many fits across the sweep); deltas are
numerically identical to the canonical estimator to ~3 dp (validated in c7).

Outputs:
    results/c8b-anticipation-per-asset.csv
    results/c8b-anticipation-summary.csv
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
from tarch_x_fast import FastTARCHX  # noqa: E402

OUT_DIR = c2.OUT_DIR
ASSETS = c2.ASSETS
REG_PRE_SWEEP = [3, 5, 7, 10]
INFRA_PRE = 3
INFRA_POST = 3
REG_POST = 3


def build_asymmetric_dummies(date_index, infra_dates, reg_dates,
                             infra_pre, infra_post, reg_pre, reg_post):
    """Aggregate D_infra / D_reg dummies with DIFFERENT pre/post windows per type."""
    idx = pd.DatetimeIndex(date_index)
    d_inf = pd.Series(0.0, index=idx, name="D_infrastructure")
    d_reg = pd.Series(0.0, index=idx, name="D_regulatory")
    for dt in infra_dates:
        dt = pd.to_datetime(dt).normalize()
        m = (idx >= dt - pd.Timedelta(days=infra_pre)) & (idx <= dt + pd.Timedelta(days=infra_post))
        d_inf.loc[m] = 1.0
    for dt in reg_dates:
        dt = pd.to_datetime(dt).normalize()
        m = (idx >= dt - pd.Timedelta(days=reg_pre)) & (idx <= dt + pd.Timedelta(days=reg_post))
        d_reg.loc[m] = 1.0
    return pd.DataFrame({"D_infrastructure": d_inf, "D_regulatory": d_reg})


def fit_deltas(returns, exog, seed=0):
    fast = FastTARCHX(returns.values, exog.values)
    p, f, ok = fast.fit_multistart(n_starts=5, seed=seed)
    # exog order: [D_infra, D_reg, S_gdelt, S_reg, S_infra] -> params[5:]
    deltas = fast.deltas(p)
    return deltas[0], deltas[1], ok  # dInfra, dReg


def main():
    print("Loading panel/sentiment/events...")
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)
    print(f"baseline: {len(inf_d)} infra, {len(reg_d)} reg")

    per_asset = []
    summary = []
    for reg_pre in REG_PRE_SWEEP:
        di_list, dr_list = [], []
        for a in ASSETS:
            r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
            dum = build_asymmetric_dummies(r.index, inf_d, reg_d,
                                           INFRA_PRE, INFRA_POST, reg_pre, REG_POST)
            s = sent.reindex(r.index).fillna(0)
            exog = pd.concat([dum[["D_infrastructure", "D_regulatory"]],
                              s[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
                             axis=1).fillna(0)
            di, dr, ok = fit_deltas(r, exog, seed=0)
            di_list.append(di); dr_list.append(dr)
            per_asset.append({"reg_pre_window": reg_pre, "asset": a,
                              "dInfra": di, "dReg": dr, "converged": ok})
        infra = np.array(di_list); reg = np.array(dr_list)
        mult = np.nanmean(infra) / np.nanmean(reg) if np.nanmean(reg) != 0 else np.nan
        t, p = stats.ttest_ind(infra, reg, equal_var=False, nan_policy="omit")
        summary.append({"reg_pre_window": reg_pre,
                        "infra_pre_window": INFRA_PRE,
                        "mean_dInfra": np.nanmean(infra), "mean_dReg": np.nanmean(reg),
                        "median_dInfra": np.nanmedian(infra), "median_dReg": np.nanmedian(reg),
                        "multiplier": mult, "welch_t": t, "welch_p": p})
        print(f"  reg_pre={reg_pre:2d}: infra={np.nanmean(infra):.3f} reg={np.nanmean(reg):.3f} "
              f"mult={mult:.2f}x  welch_p={p:.4f}")

    pd.DataFrame(per_asset).to_csv(OUT_DIR / "c8b-anticipation-per-asset.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT_DIR / "c8b-anticipation-summary.csv", index=False)
    print("\nSaved: c8b-anticipation-per-asset.csv, c8b-anticipation-summary.csv")
    print("Reference: reg_pre=3 is the symmetric published spec (~3.49x).")


if __name__ == "__main__":
    main()
