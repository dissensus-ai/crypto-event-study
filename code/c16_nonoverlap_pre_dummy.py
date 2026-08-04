"""
C16: Non-overlapping pre-event dummy -- a clean anticipation test for BOTH legs.
================================================================================

Two referee points on the anticipation analysis:

 (#3, Major) Widening the regulatory pre-window (c8b) mechanically DILUTES the
   average dummy: the extra quiet pre-days lower the mean D_reg coefficient, so a
   *falling* coefficient is exactly what dilution produces and cannot by itself
   rule out concentrated anticipation. The clean test is a SEPARATE,
   NON-OVERLAPPING pre-event dummy that isolates any pre-trend without
   contaminating it with the event-window days.

 (#4, Minor) c8b lengthened only the regulatory pre-window; infrastructure was
   held at [-3,+3]. But infrastructure events can also leak (on-chain rumours, a
   whale draining liquidity before an exploit is public). A symmetric treatment
   is warranted.

This script addresses both at once. For each asset we fit ONE GJR-GARCH-X with
FOUR event regressors instead of two:

    D_infra_event  : infrastructure [-3, +3]   (the main event window)
    D_reg_event    : regulatory     [-3, +3]
    D_infra_pre    : infrastructure [-10, -4]   (non-overlapping pre-window)
    D_reg_pre      : regulatory     [-10, -4]

plus the three sentiment controls, exactly as in the baseline model. Because the
pre-window [-10,-4] does not overlap the event window [-3,+3], the pre-dummy
coefficient d_pre measures elevated conditional variance in the run-up that the
symmetric window misses. If d_pre is small and not systematically positive, there
is no concentrated pre-event variance to absorb -- for EITHER leg -- and the
symmetric-window multiplier is not an anticipation artefact.

Point estimates use FastTARCHX (numerically identical to the canonical estimator
to ~3dp, validated in c7). The cross-asset-robust copula bootstrap remains the
inference of record for the headline multiplier; the naive one-sample t-tests
here are descriptive, flagged as such.

Outputs (event-study/r1-revision/):
    c16-nonoverlap-pre-per-asset.csv
    c16-nonoverlap-pre-summary.csv
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
import c2_relaxed_threshold_sensitivity as c2   # noqa: E402
from tarch_x_fast import FastTARCHX             # noqa: E402

OUT_DIR = c2.OUT_DIR
ASSETS = c2.ASSETS
EVENT_PRE, EVENT_POST = 3, 3
PRE_START, PRE_END = 10, 4   # non-overlapping pre-window [-10, -4]


def window_dummy(idx, dates, days_before, days_after, name):
    """Binary dummy = 1 on [date - days_before, date - days_after] (inclusive).

    For an event window use (days_before=3, days_after=-3) i.e. [-3,+3]; for the
    pre-window use (days_before=10, days_after=4) i.e. [-10,-4]. `days_after` is
    the offset of the *late* edge measured as days BEFORE the date (so +3 after
    the date is days_after=-3).
    """
    idx = pd.DatetimeIndex(idx)
    d = pd.Series(0.0, index=idx, name=name)
    for dt in dates:
        dt = pd.to_datetime(dt).normalize()
        lo = dt - pd.Timedelta(days=days_before)
        hi = dt - pd.Timedelta(days=days_after)
        d.loc[(idx >= lo) & (idx <= hi)] = 1.0
    return d


def main():
    print("Loading panel/sentiment/events...")
    panel = c2.load_returns_panel()
    common = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in panel.values()])))
    sent = c2.load_sentiment_daily(common)
    events = pd.read_csv(c2.DATA_DIR / "events.csv"); events["date"] = pd.to_datetime(events["date"])
    census = pd.read_csv(OUT_DIR / "c1-dropout-census.csv"); census["date"] = pd.to_datetime(census["date"])
    inf_d, reg_d = c2.get_event_dates_for_spec("S1_baseline", events, census)
    print(f"baseline: {len(inf_d)} infra, {len(reg_d)} reg")

    rows = []
    for a in ASSETS:
        r = panel[a].loc[panel[a].index >= pd.Timestamp(c2.START_DATE)]
        idx = r.index
        d_inf_ev = window_dummy(idx, inf_d, EVENT_PRE, -EVENT_POST, "D_infra_event")
        d_reg_ev = window_dummy(idx, reg_d, EVENT_PRE, -EVENT_POST, "D_reg_event")
        d_inf_pre = window_dummy(idx, inf_d, PRE_START, PRE_END, "D_infra_pre")
        d_reg_pre = window_dummy(idx, reg_d, PRE_START, PRE_END, "D_reg_pre")
        s = sent.reindex(idx).fillna(0)
        exog = pd.concat(
            [d_inf_ev, d_reg_ev, d_inf_pre, d_reg_pre,
             s[["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]]],
            axis=1).fillna(0)
        fast = FastTARCHX(r.values, exog.values)
        p, f, ok = fast.fit_multistart(n_starts=6, seed=0)
        de = fast.deltas(p)  # [d_inf_ev, d_reg_ev, d_inf_pre, d_reg_pre, s...]
        rows.append({
            "asset": a,
            "d_infra_event": de[0], "d_reg_event": de[1],
            "d_infra_pre": de[2], "d_reg_pre": de[3],
            "converged": ok,
        })
        print(f"  {a}: infra_ev={de[0]:+.3f} reg_ev={de[1]:+.3f} | "
              f"infra_pre={de[2]:+.3f} reg_pre={de[3]:+.3f}  [ok={ok}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c16-nonoverlap-pre-per-asset.csv", index=False)

    # cross-asset means + naive one-sample t-tests (descriptive)
    def summ(col):
        v = df[col].values.astype(float)
        t, pv = stats.ttest_1samp(v, 0.0)
        return np.nanmean(v), np.nanmedian(v), float(t), float(pv)

    summary_rows = []
    for col, lab in [("d_infra_event", "infra event [-3,+3]"),
                     ("d_reg_event", "reg event [-3,+3]"),
                     ("d_infra_pre", "infra pre [-10,-4]"),
                     ("d_reg_pre", "reg pre [-10,-4]")]:
        m, md, t, pv = summ(col)
        summary_rows.append({"coefficient": lab, "cross_asset_mean": m,
                             "median": md, "naive_t_vs0": t, "naive_p_vs0": pv})
    sm = pd.DataFrame(summary_rows)
    ev_mult = (df["d_infra_event"].mean() / df["d_reg_event"].mean()
               if df["d_reg_event"].mean() != 0 else np.nan)
    sm_extra = pd.DataFrame([{"coefficient": "event-window multiplier (infra/reg)",
                              "cross_asset_mean": ev_mult, "median": np.nan,
                              "naive_t_vs0": np.nan, "naive_p_vs0": np.nan}])
    sm = pd.concat([sm, sm_extra], ignore_index=True)
    sm.to_csv(OUT_DIR / "c16-nonoverlap-pre-summary.csv", index=False)

    print("\n" + "=" * 70)
    print("CROSS-ASSET SUMMARY (naive t vs 0 is descriptive only)")
    print("=" * 70)
    for _, rr in sm.iterrows():
        if pd.isna(rr["naive_p_vs0"]):
            print(f"  {rr['coefficient']:34s} mean={rr['cross_asset_mean']:+.3f}")
        else:
            print(f"  {rr['coefficient']:34s} mean={rr['cross_asset_mean']:+.3f} "
                  f"median={rr['median']:+.3f}  naive t={rr['naive_t_vs0']:+.2f} p={rr['naive_p_vs0']:.3f}")
    print("\nRead: a small, non-significant d_*_pre means no concentrated pre-event")
    print("variance the symmetric window misses -- for either leg.")
    print("Saved: c16-nonoverlap-pre-per-asset.csv, c16-nonoverlap-pre-summary.csv")


if __name__ == "__main__":
    main()
