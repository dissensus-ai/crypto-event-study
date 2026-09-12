"""
C8e: Sub-sample sizes + persistence standard errors.
====================================================

The paper reports a descriptive within-regime persistence drop from a symmetric
GARCH(1,1) fit. This script is an exploratory companion diagnostic: it fits a
different event-free GJR-GARCH specification to the same exact segment partition
and reports its Hessian-based uncertainty. It does not provide like-for-like
standard errors for the plain-GARCH persistence values tabulated in the paper.

This refits a plain GJR-GARCH (Student-t, NO event/sentiment exog) on each regime
segment defined by c3's absolute-return breaks (inclusive segment endpoints are
taken from c3-subsample-persistence.csv), using the canonical TARCHXEstimator so
we get model standard errors. For each (asset, segment) it reports:
    n_obs, alpha (+/-SE), beta (+/-SE), gamma (+/-SE),
    signed GJR persistence functional P = alpha + beta + gamma/2 under symmetric
        innovations, with a delta-method SE (the estimator's constraint instead
        uses alpha + beta + |gamma|/2),
    a small-sample flag (n_obs < 500).

It then summarizes the exploratory GJR differences by segment size. It does not
adjudicate the paper's descriptive plain-GARCH contrast.

Outputs:
    results/c8e-persistence-se.csv
"""
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_relaxed_threshold_sensitivity as c2  # noqa: E402

OUT_DIR = c2.OUT_DIR
ASSETS = c2.ASSETS
SMALL_N = 500


def persistence_and_se(res):
    """P = alpha + beta + gamma/2; SE via delta method using the param SEs.

    The canonical estimator returns only marginal SEs (diag of cov), not the full
    covariance matrix. We therefore report a working independent-sum approximation:
        Var(P) ~ Var(a) + Var(b) + Var(g)/4
    This ignores cross-covariances and is not guaranteed to be conservative. It is
    retained as an exploratory GJR diagnostic, not inference for c3's plain-GARCH
    persistence contrast.
    """
    a = res.params.get("alpha", np.nan)
    b = res.params.get("beta", np.nan)
    g = res.params.get("gamma", np.nan)
    sa = res.std_errors.get("alpha", np.nan)
    sb = res.std_errors.get("beta", np.nan)
    sg = res.std_errors.get("gamma", np.nan)
    P = a + b + g / 2.0
    varP = (sa ** 2 if np.isfinite(sa) else np.nan) \
        + (sb ** 2 if np.isfinite(sb) else np.nan) \
        + (sg ** 2 / 4.0 if np.isfinite(sg) else np.nan)
    seP = np.sqrt(varP) if np.isfinite(varP) else np.nan
    return a, sa, b, sb, g, sg, P, seP


def main():
    print("Loading panel + c3 segment boundaries...")
    panel = c2.load_returns_panel()
    seg = pd.read_csv(OUT_DIR / "c3-subsample-persistence.csv")
    seg["subsample_start"] = pd.to_datetime(seg["subsample_start"])
    seg["subsample_end"] = pd.to_datetime(seg["subsample_end"])

    rows = []
    for a in ASSETS:
        r = panel[a].dropna()
        # full-sample fit (no exog) for the reference persistence + SE
        full = c2.TARCHXEstimator(r.loc[r.index >= pd.Timestamp(c2.START_DATE)], None)
        res_full = full.estimate(method="SLSQP", max_iter=2000)
        fa, fsa, fb, fsb, fg, fsg, fP, fseP = persistence_and_se(res_full)
        print(f"\n{a}: FULL n={len(r)} P={fP:.4f} (+/-{fseP:.4f}) [a={fa:.3f} b={fb:.3f} g={fg:.3f}]")
        rows.append({"asset": a, "segment": "FULL", "start": r.index.min().date(),
                     "end": r.index.max().date(), "n_obs": len(r),
                     "alpha": fa, "se_alpha": fsa, "beta": fb, "se_beta": fsb,
                     "gamma": fg, "se_gamma": fsg, "persistence": fP, "se_persistence": fseP,
                     "small_sample_n_lt_500": len(r) < SMALL_N})

        segs = seg[seg["asset"] == a].reset_index(drop=True)
        covered = []
        for _, srow in segs.iterrows():
            lo, hi = srow["subsample_start"], srow["subsample_end"]
            covered.append(r.loc[(r.index >= lo) & (r.index <= hi)].index.asi8)
        covered_ns = np.concatenate(covered) if covered else np.asarray([], dtype=np.int64)
        if not np.array_equal(covered_ns, r.index.asi8):
            raise RuntimeError(
                f"c3 segment endpoints do not partition the {a} return series exactly"
            )
        for i, srow in segs.iterrows():
            lo, hi = srow["subsample_start"], srow["subsample_end"]
            rs = r.loc[(r.index >= lo) & (r.index <= hi)]
            n = len(rs)
            if n < 50:
                print(f"   seg{i} [{lo.date()}..{hi.date()}] n={n} TOO SMALL, skipped")
                continue
            est = c2.TARCHXEstimator(rs, None)
            res = est.estimate(method="SLSQP", max_iter=2000)
            sa_, ssa, sb_, ssb, sg_, ssg, sP, sseP = persistence_and_se(res)
            flag = n < SMALL_N
            print(f"   seg{i} [{lo.date()}..{hi.date()}] n={n}{' (SMALL)' if flag else ''} "
                  f"P={sP:.4f} (+/-{sseP:.4f}) [a={sa_:.3f} b={sb_:.3f} g={sg_:.3f}]")
            rows.append({"asset": a, "segment": f"seg{i}", "start": lo.date(), "end": hi.date(),
                         "n_obs": n, "alpha": sa_, "se_alpha": ssa, "beta": sb_, "se_beta": ssb,
                         "gamma": sg_, "se_gamma": ssg, "persistence": sP, "se_persistence": sseP,
                         "small_sample_n_lt_500": flag,
                         "full_persistence": fP, "drop_vs_full": fP - sP,
                         "drop_exceeds_2se": (abs(fP - sP) > 2 * sseP) if np.isfinite(sseP) else np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "c8e-persistence-se.csv", index=False)

    # verdict on the drop
    subs = df[df["segment"] != "FULL"].copy()
    big = subs[subs["n_obs"] >= SMALL_N]
    small = subs[subs["n_obs"] < SMALL_N]
    print(f"\n=== Exploratory GJR persistence diagnostic ===")
    print(f"  sub-segments: {len(subs)} total, {len(small)} with n<500 (small-sample flagged)")
    if len(big):
        print(f"  large (n>=500) sub-segments mean persistence: {big['persistence'].mean():.4f} "
              f"(mean drop vs full {big['drop_vs_full'].mean():+.4f})")
    if len(small):
        print(f"  small (n<500)  sub-segments mean persistence: {small['persistence'].mean():.4f} "
              f"(mean drop vs full {small['drop_vs_full'].mean():+.4f})")
    n_robust = subs["drop_exceeds_2se"].sum()
    print(f"  sub-segments whose drop exceeds 2*SE: {n_robust}/{len(subs)}")
    print("Saved: c8e-persistence-se.csv")


if __name__ == "__main__":
    main()
