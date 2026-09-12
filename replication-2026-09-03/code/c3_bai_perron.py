"""
C3: Bai-Perron-Style Structural-Break Tests
============================================

For each of the six cryptocurrencies, detect multiple structural breaks in
two volatility-proxy series:
  (a) absolute daily log returns (|r_t|, model-free variance proxy)
  (b) the conditional-variance series from a GARCH(1,1) fit

We use the PELT algorithm (ruptures) with an L2 cost function for
change-point detection. Over a penalty grid, we choose the candidate with
minimum BIC among those with at most five breaks, enforcing
``min_size=max(floor(0.15 N), 30)``. For each detected break we report
the date, pre/post mean proxy levels, and a t-statistic on the
difference. Break confidence intervals are approximated by bootstrapping
the change-point location (200 reps). The per-break prefix-versus-suffix
statistics and single-split confidence intervals are post-selection,
nonlocal descriptive diagnostics; they are not used for manuscript inference.

We then re-estimate the symmetric GARCH(1,1) baseline (no events, no sentiment) on
each sub-sample defined by the breaks, and report whether α + β drops from
the corresponding asset's full-sample value (0.954--0.998 in this sample).

Outputs:
    results/c3-bai-perron-results.csv
    results/c3-subsample-persistence.csv
    results/c3-break-summary.md
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
# Data location: prefer a data/ directory beside this file (flat replication
# package layout); otherwise fall back to <root>/code/data (working-tree layout).
CODE_DIR = _CD if (_CD := Path(__file__).resolve().parent).joinpath("data").is_dir() else ROOT / "code"
DATA_DIR = CODE_DIR / "data"
# Output location. Works in both layouts: the working tree (<root>/r1-revision/code/)
# writes to <root>/r1-revision/, while a flat replication package (<pkg>/code/)
# writes to <pkg>/results/. Override with CES_OUT_DIR.
import os as _os
_here = Path(__file__).resolve().parent
OUT_DIR = Path(_os.environ.get("CES_OUT_DIR",
               _here.parent if _here.parent.name == "r1-revision" else _here.parent / "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_relaxed_threshold_sensitivity import load_returns_panel  # noqa: E402

ASSETS = ["btc", "eth", "xrp", "bnb", "ltc", "ada"]
TRIMMING = 0.15  # Bai-Perron-style minimum segment fraction
MAX_BREAKS = 5
BOOTSTRAP_REPS = 200


# -----------------------------------------------------------------------------
# GARCH(1,1) fit (used both for full sample and sub-sample re-estimation)
# -----------------------------------------------------------------------------
def fit_garch11(returns: pd.Series, max_iter: int = 2000):
    """Fit GARCH(1,1) with Student-t innovations using maximum likelihood.

    Returns (params dict, conditional_variance series).
    """
    from scipy.optimize import minimize

    r = returns.dropna().values
    n = len(r)
    if n < 100:
        return None, None

    def neg_ll(params):
        omega, alpha, beta, nu = params
        # Stationarity / domain
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.999 or nu <= 2.1 or nu >= 50:
            return 1e10
        mu = r.mean()
        eps = r - mu
        var = np.zeros(n)
        var[0] = np.var(r)
        for t in range(1, n):
            var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
            if var[t] <= 0:
                return 1e10
        # Student-t log likelihood
        from scipy.special import gammaln
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
              - 0.5 * np.log(np.pi * (nu - 2))
              - 0.5 * np.log(var)
              - 0.5 * (nu + 1) * np.log(1 + eps ** 2 / (var * (nu - 2))))
        return -float(ll.sum())

    x0 = np.array([np.var(r) * 0.05, 0.06, 0.92, 5.0])
    bounds = [(1e-8, None), (1e-8, 0.3), (1e-8, 0.98), (2.1, 50)]
    cons = [{"type": "ineq", "fun": lambda x: 0.998 - (x[1] + x[2])}]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": max_iter})
    except Exception:
        return None, None

    if not res.success or res.fun >= 1e9:
        return None, None
    omega, alpha, beta, nu = res.x
    mu = r.mean()
    eps = r - mu
    var = np.zeros(n)
    var[0] = np.var(r)
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
    params = {"omega": omega, "alpha": alpha, "beta": beta,
              "nu": nu, "persistence": alpha + beta, "log_lik": -res.fun, "n_obs": n}
    return params, pd.Series(np.sqrt(var), index=returns.dropna().index)


# -----------------------------------------------------------------------------
# Break detection via PELT with L2 cost on the variance-proxy series
# -----------------------------------------------------------------------------
def detect_breaks(series: pd.Series, max_breaks: int = MAX_BREAKS,
                  min_size_frac: float = TRIMMING):
    """PELT change-point detection with BIC-penalised model selection.

    Returns list of break indices (positions in the series, NOT including 0 or n).
    """
    y = series.dropna().values.astype(float)
    n = len(y)
    if n < 100:
        return []
    min_size = max(int(min_size_frac * n), 30)

    # Calibrate penalty across a range; pick the smallest k breaks <= max_breaks
    # where BIC improvement saturates.
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(y)

    # Sweep penalties from small to large; record (n_breaks, ssr) curve.
    penalties = np.logspace(-2, 4, 60)
    candidates = []
    for pen in penalties:
        try:
            bkps = algo.predict(pen=pen)
        except Exception:
            continue
        n_breaks = len(bkps) - 1
        if n_breaks > max_breaks or n_breaks < 0:
            continue
        # Compute SSR for this segmentation
        ssr = 0.0
        prev = 0
        for b in bkps:
            seg = y[prev:b]
            ssr += float(np.sum((seg - seg.mean()) ** 2))
            prev = b
        bic = n * np.log(ssr / n + 1e-12) + (n_breaks + 1) * np.log(n)
        candidates.append((n_breaks, bic, bkps))

    if not candidates:
        return []

    # Pick segmentation that minimises BIC
    best = min(candidates, key=lambda c: c[1])
    bkps = best[2]
    # Remove the trailing endpoint (n)
    return [b for b in bkps if 0 < b < n]


def bootstrap_break_ci(series: pd.Series, break_idx: int,
                       reps: int = BOOTSTRAP_REPS, alpha: float = 0.10) -> tuple:
    """Bootstrap the break-point location for a 90% CI.

    Resample the residual differences around the break and re-detect on each
    resample. Returns (lower_idx, upper_idx).
    """
    y = series.dropna().values.astype(float)
    n = len(y)
    pre = y[:break_idx]
    post = y[break_idx:]
    pre_mean, post_mean = pre.mean(), post.mean()
    pre_res = pre - pre_mean
    post_res = post - post_mean

    locs = []
    rng = np.random.default_rng(42)
    for _ in range(reps):
        rs_pre = rng.choice(pre_res, size=len(pre_res), replace=True) + pre_mean
        rs_post = rng.choice(post_res, size=len(post_res), replace=True) + post_mean
        ys = np.concatenate([rs_pre, rs_post])
        # Find best single-break location via min-SSR sweep around true break
        radius = max(int(0.1 * n), 30)
        lo = max(1, break_idx - radius)
        hi = min(n - 1, break_idx + radius)
        best_b, best_ssr = break_idx, np.inf
        for b in range(lo, hi):
            s1 = ys[:b]
            s2 = ys[b:]
            ssr = float(np.sum((s1 - s1.mean()) ** 2) + np.sum((s2 - s2.mean()) ** 2))
            if ssr < best_ssr:
                best_ssr = ssr
                best_b = b
        locs.append(best_b)
    locs = np.array(locs)
    lo = int(np.quantile(locs, alpha / 2))
    hi = int(np.quantile(locs, 1 - alpha / 2))
    return lo, hi


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def main():
    print("Loading returns panel...")
    panel = load_returns_panel()

    results = []
    persistence_rows = []

    for a in ASSETS:
        r = panel[a]
        print(f"\n=== {a.upper()}: n={len(r)} ===")
        # Series 1: absolute returns
        abs_r = r.abs()
        # Series 2: conditional variance from GARCH(1,1) fit
        full_params, full_vol = fit_garch11(r)
        if full_params is None:
            print(f"  [WARN] GARCH(1,1) fit failed for {a}")
            cond_var = pd.Series(dtype=float)
            full_persistence = np.nan
        else:
            cond_var = (full_vol ** 2)
            full_persistence = full_params["persistence"]
            print(f"  Full sample: α+β = {full_persistence:.4f}")

        for series_type, series in [("abs_return", abs_r), ("cond_variance", cond_var)]:
            if len(series.dropna()) < 100:
                continue
            print(f"  Detecting breaks in {series_type}...")
            break_idxs = detect_breaks(series)
            print(f"    -> {len(break_idxs)} break(s) at positions {break_idxs}")

            dates = series.dropna().index
            for bi, b_idx in enumerate(break_idxs):
                if b_idx >= len(dates):
                    continue
                break_date = dates[b_idx]
                y = series.dropna().values
                pre = y[:b_idx]
                post = y[b_idx:]
                # t-test (Welch) on the mean difference
                t_stat, p_val = stats.ttest_ind(post, pre, equal_var=False)
                # CI on break location (90%)
                lo_i, hi_i = bootstrap_break_ci(series, b_idx)
                ci_lo_date = dates[lo_i] if 0 <= lo_i < len(dates) else dates[0]
                ci_hi_date = dates[hi_i] if 0 <= hi_i < len(dates) else dates[-1]

                results.append({
                    "asset": a,
                    "series_type": series_type,
                    "break_number": bi + 1,
                    "break_date": break_date.strftime("%Y-%m-%d"),
                    "ci_lo_date": ci_lo_date.strftime("%Y-%m-%d"),
                    "ci_hi_date": ci_hi_date.strftime("%Y-%m-%d"),
                    "pre_mean": float(np.mean(pre)),
                    "post_mean": float(np.mean(post)),
                    "delta_mean": float(np.mean(post) - np.mean(pre)),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "pre_n": int(len(pre)),
                    "post_n": int(len(post)),
                })

        # ---------- Sub-sample re-estimation of GARCH(1,1) persistence ----------
        # Use the break points from absolute returns to define sub-samples.
        abs_breaks = [r for r in results if r["asset"] == a and r["series_type"] == "abs_return"]
        break_dates = sorted(pd.to_datetime([b["break_date"] for b in abs_breaks]))
        break_pos = [int(r.index.get_loc(date)) for date in break_dates]
        bounds = [0, *break_pos, len(r)]
        for i, (lo_pos, hi_pos) in enumerate(zip(bounds[:-1], bounds[1:])):
            # Ruptures reports each breakpoint as the exclusive end of the
            # preceding segment. Positional half-open slices therefore form an
            # exact partition: no break-date observation is duplicated, and
            # the final observation is retained. The CSV stores each slice's
            # actual first/last observation dates as inclusive endpoints.
            sub = r.iloc[lo_pos:hi_pos]
            if len(sub) < 100:
                continue
            sp, _ = fit_garch11(sub)
            if sp is None:
                continue
            persistence_rows.append({
                "asset": a,
                "subsample_start": sub.index[0].strftime("%Y-%m-%d"),
                "subsample_end": sub.index[-1].strftime("%Y-%m-%d"),
                "n_obs": int(sp["n_obs"]),
                "alpha": sp["alpha"],
                "beta": sp["beta"],
                "persistence": sp["persistence"],
                "log_lik": sp["log_lik"],
                "full_sample_persistence": full_persistence,
            })

    df = pd.DataFrame(results)
    df_persist = pd.DataFrame(persistence_rows)
    df.to_csv(OUT_DIR / "c3-bai-perron-results.csv", index=False)
    df_persist.to_csv(OUT_DIR / "c3-subsample-persistence.csv", index=False)
    print(f"\nWrote {OUT_DIR/'c3-bai-perron-results.csv'} ({len(df)} breaks)")
    print(f"Wrote {OUT_DIR/'c3-subsample-persistence.csv'} ({len(df_persist)} sub-samples)")

    # ---------------- Markdown summary ----------------
    lines = ["# C3 — PELT/BIC Structural-Break Diagnostics "
             "(Bai-Perron-Style Trimming): Summary\n"]
    lines.append(
        "Multiple structural-break detection on two variance-proxy series per asset: "
        "(a) |r_t|, the model-free variance proxy, and (b) the conditional-variance "
        "series from a GARCH(1,1) fit. Break-point detection via PELT (ruptures, L2 "
        "cost) with BIC-penalised model selection. Bai-Perron-style trimming fraction "
        f"of 0.15 enforced (`min_size`), max {MAX_BREAKS} breaks per series.\n"
    )
    lines.append(
        "The per-break t/p columns compare the entire prefix with the entire suffix, "
        "and each 90% interval bootstraps a single split after the multi-break model "
        "has been selected. They are approximate, post-selection, nonlocal descriptive "
        "diagnostics and are not used for manuscript inference.\n"
    )

    lines.append("## Detected Breaks (per asset, per series)\n")
    lines.append("| Asset | Series | # | Date | 90% CI | Pre mean | Post mean | Δ | t-stat | p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['asset'].upper()} | {r['series_type']} | {r['break_number']} | "
            f"{r['break_date']} | [{r['ci_lo_date']}, {r['ci_hi_date']}] | "
            f"{r['pre_mean']:.4f} | {r['post_mean']:.4f} | "
            f"{r['delta_mean']:+.4f} | {r['t_stat']:+.2f} | {r['p_value']:.4f} |"
        )

    # Persistence sub-sample table
    lines.append("\n## Sub-Sample GARCH(1,1) Persistence (α + β) by Regime\n")
    lines.append("| Asset | Sub-sample | n | α | β | α + β | Full-sample α + β |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df_persist.iterrows():
        lines.append(
            f"| {r['asset'].upper()} | {r['subsample_start']} → {r['subsample_end']} | "
            f"{int(r['n_obs'])} | {r['alpha']:.4f} | {r['beta']:.4f} | "
            f"**{r['persistence']:.4f}** | {r['full_sample_persistence']:.4f} |"
        )

    # Interpretation
    lines.append("\n## Interpretation\n")
    # Mean sub-sample persistence vs full-sample
    if not df_persist.empty:
        # Equal-weight assets, not segments or distinct numerical values.
        mean_sub = df_persist.groupby("asset")["persistence"].mean().mean()
        mean_full = df_persist.groupby("asset")["full_sample_persistence"].first().mean()
        diff = mean_full - mean_sub
        lines.append(
            f"- Equal-weight mean full-sample persistence (α + β): **{mean_full:.4f}**.\n"
            f"- Equal-weight mean of the six assets' within-segment averages: "
            f"**{mean_sub:.4f}**.\n"
            f"- Difference: **{diff:+.4f}** (within-segment point estimates lower).\n"
        )
        if diff > 0.01:
            lines.append(
                "The within-segment point estimates are lower on average. This is "
                "consistent with structural shifts contributing to full-sample "
                "persistence, but short segments, break selection, and wide parameter "
                "uncertainty prevent treating the difference as a resolved structural fact.\n"
            )
        else:
            lines.append(
                "The within-sub-sample persistence remains near unity, suggesting the "
                "near-integration is not attenuated by this segmentation. This remains "
                "a descriptive comparison rather than a formal break-adjusted test.\n"
            )

    # Crisis-alignment check
    lines.append("\n## Crisis-Regime Alignment\n")
    known_crises = [
        ("COVID-19 shock", "2020-03-12"),
        ("Terra/UST collapse", "2022-05-09"),
        ("FTX bankruptcy", "2022-11-11"),
        ("SVB / USDC depeg", "2023-03-10"),
        ("Bybit hack", "2025-02-21"),
    ]
    lines.append("Aligning detected breaks (|r_t| series) to known crisis dates "
                 "within ±60 days:\n")
    lines.append("| Crisis | Date | Aligned breaks |")
    lines.append("|---|---|---|")
    abs_df = df[df["series_type"] == "abs_return"].copy()
    abs_df["break_date_dt"] = pd.to_datetime(abs_df["break_date"])
    for name, dstr in known_crises:
        d = pd.to_datetime(dstr)
        matched = abs_df[(abs_df["break_date_dt"] - d).abs() <= pd.Timedelta(days=60)]
        if matched.empty:
            lines.append(f"| {name} | {dstr} | (none) |")
        else:
            agg = ", ".join(f"{r['asset'].upper()}@{r['break_date']}" for _, r in matched.iterrows())
            lines.append(f"| {name} | {dstr} | {agg} |")

    lines.append("\n## Files\n")
    lines.append("- Per-break results: `c3-bai-perron-results.csv`")
    lines.append("- Sub-sample persistence: `c3-subsample-persistence.csv`\n")

    (OUT_DIR / "c3-break-summary.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR/'c3-break-summary.md'}")


if __name__ == "__main__":
    main()
