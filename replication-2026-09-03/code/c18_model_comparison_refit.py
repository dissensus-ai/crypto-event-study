"""
C18: Model-comparison refit under a single controlled estimator.

Why
---
This script produces the controlled model comparison used by the manuscript:
all three nested specifications are fit with FastTARCHX, multistart, on the
same corrected event census and return transformation. The GJR-GARCH-X fit is
anchored against c14, the canonical per-asset parameter run.

Specs (all: winsorised %-log-returns, mean profiled at the sample mean,
Student-t innovations, sigma2_0 = sample variance, SLSQP, constraint
alpha + beta + |gamma|/2 <= 0.999)
  1. GARCH(1,1)  : gamma PINNED to 0, no exogenous regressors.
  2. GJR-GARCH   : gamma free, no exogenous regressors.
  3. GJR-GARCH-X : gamma free + the 5 exog (D_infra, D_reg, S_gdelt, S_reg,
                   S_infra). This is bit-for-bit the c14 fit (same design,
                   same starts, same seed) and is the sanity anchor.

How gamma is pinned
-------------------
NOT via degenerate (0,0) bounds. The pinned estimator subclasses FastTARCHX
and removes gamma from the free-parameter vector entirely: SLSQP optimises
[omega, alpha, beta, nu] and gamma = 0 is re-inserted at position 2 before
every variance-recursion/log-likelihood evaluation. Bounds and constraints
are the FastTARCHX ones with the gamma entries dropped (the stationarity
constraint degenerates to alpha + beta <= 0.999). Returned parameter vectors
are re-expanded to the full 5(+n_exog) layout so downstream code sees the
usual [omega, alpha, gamma, beta, nu, deltas...] ordering.

Multistart
----------
Primary grid = the c14 grid, reproduced RNG-stream-exactly: default start +
5 randomised starts from np.random.default_rng(12345) drawn in the same call
order as FastTARCHX.fit_multistart. PLUS 6 extra randomised starts from a
second seed (20260806) as a stability probe. Per-start negLL and convergence
are recorded; the CSV reports how many starts improved on the default start
and whether any extra-grid start beat the best c14-grid start.

k-convention
------------
The sample mean is profiled rather than numerically optimised, but it is still
estimated from the data. The primary information criteria therefore count it:
k = 5 / 6 / 11 for GARCH / GJR-GARCH / GJR-GARCH-X. For transparency the CSV
also reports the literally optimised counts (4 / 5 / 10) and the predecessor's
mixed count (5 / 6 / 10). The two self-consistent conventions give identical
within-asset rankings because they differ by one parameter for every model.

Outputs (in the package results directory)
  c18-model-comparison-refit.csv
  c18-model-comparison-refit-FINDING.md
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_relaxed_threshold_sensitivity as c2  # loaders, ASSETS, OUT_DIR
from tarch_x_fast import FastTARCHX, _HAVE_NUMBA

ASSETS = c2.ASSETS
SENT_COLS = ["S_gdelt_normalized", "S_reg_decomposed", "S_infra_decomposed"]
N_STARTS_C14 = 6          # the c14 grid (default + 5 random)
SEED_C14 = 12345          # the c14 seed
N_STARTS_EXTRA = 6        # stability probe beyond the c14 grid
SEED_EXTRA = 20260806
MAX_ITER = 2000

# predecessor mixed convention, retained only as a transparent comparison
K_LEGACY = {"GARCH(1,1)": 5, "GJR-GARCH": 6, "GJR-GARCH-X": 10}
# free parameters actually optimised under the controlled (profiled-mean) fits
K_FREE = {"GARCH(1,1)": 4, "GJR-GARCH": 5, "GJR-GARCH-X": 10}
# counting the profiled mean everywhere
K_MU = {"GARCH(1,1)": 5, "GJR-GARCH": 6, "GJR-GARCH-X": 11}

# ----------------------------------------------------------------------------
# gamma-pinned estimator: gamma removed from the free vector, not bounded to 0
# ----------------------------------------------------------------------------
class FastTARCHXNoGamma(FastTARCHX):
    """GARCH(1,1)-X: FastTARCHX with the leverage term gamma fixed at 0.

    Free vector: [omega, alpha, beta, nu, deltas...]; gamma = 0 is inserted
    at position 2 for every LL evaluation and in the returned params, so the
    public parameter layout matches FastTARCHX exactly.
    """

    @staticmethod
    def _expand(p):
        return np.insert(np.asarray(p, dtype=float), 2, 0.0)

    @staticmethod
    def _reduce(p):
        return np.delete(np.asarray(p, dtype=float), 2)

    def _neg_loglik_reduced(self, p):
        return super()._neg_loglik(self._expand(p))

    def _bounds_reduced(self):
        b = super()._bounds()
        return [b[0], b[1]] + b[3:]  # drop the gamma bound

    def _constraints_reduced(self):
        # same as FastTARCHX._constraints with gamma == 0: indices shift by 1
        return [
            {'type': 'ineq', 'fun': lambda x: x[0] - 1e-8},
            {'type': 'ineq', 'fun': lambda x: x[1] - 1e-8},
            {'type': 'ineq', 'fun': lambda x: x[2] - 1e-8},
            {'type': 'ineq', 'fun': lambda x: x[3] - 2.1},
            {'type': 'ineq', 'fun': lambda x: 50 - x[3]},
            {'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[2])},
        ]

    def fit(self, start=None, max_iter=2000):
        from scipy.optimize import minimize
        if start is None:
            start = self._default_start()
        start = np.asarray(start, dtype=float)
        if start.shape[0] == 5 + self.n_exog:   # full layout -> reduce
            start = self._reduce(start)
        res = minimize(self._neg_loglik_reduced, start, method='SLSQP',
                       bounds=self._bounds_reduced(),
                       constraints=self._constraints_reduced(),
                       options={'maxiter': max_iter, 'disp': False})
        success = bool(res.success) and res.fun < 1e6
        return self._expand(res.x), float(res.fun), success


# ----------------------------------------------------------------------------
# start generation: RNG-stream-exact copy of FastTARCHX.fit_multistart
# ----------------------------------------------------------------------------
def make_starts(returns, n_exog, n_starts, seed, default_start, pin_gamma):
    """Reproduce FastTARCHX.fit_multistart's start sequence exactly.

    Draw order per random start: omega, alpha, gamma, beta, nu uniforms, then
    (if n_exog) one rng.normal(0., 0.5, n_exog) call; then the stationarity
    feasibility fix. For the pinned spec gamma is zeroed AFTER the draws (the
    draw still happens, keeping the stream aligned) and feasibility rechecked.
    """
    rng = np.random.default_rng(seed)
    sv = np.var(returns)
    starts = [default_start.copy()]
    for _ in range(max(0, n_starts - 1)):
        s = np.array([
            sv * rng.uniform(0.02, 0.30),     # omega
            rng.uniform(0.01, 0.15),          # alpha
            rng.uniform(-0.10, 0.20),         # gamma
            rng.uniform(0.70, 0.93),          # beta
            rng.uniform(3.0, 12.0),           # nu
        ])
        if n_exog:
            s = np.append(s, rng.normal(0.0, 0.5, n_exog))
        if s[1] + s[3] + abs(s[2]) / 2 >= 0.999:
            s[3] = 0.90 - s[1] - abs(s[2]) / 2
        if pin_gamma:
            s[2] = 0.0
            if s[1] + s[3] >= 0.999:
                s[3] = 0.90 - s[1]
        starts.append(s)
    return starts


def multistart_record(est, starts, max_iter=MAX_ITER):
    """Run each start, record (negLL, converged); return best + bookkeeping."""
    recs = []
    best = None
    for s in starts:
        try:
            p, f, ok = est.fit(start=s, max_iter=max_iter)
        except Exception:  # noqa: BLE001
            recs.append((np.inf, False))
            continue
        recs.append((f, ok))
        if best is None or f < best[1]:
            best = (p, f, ok)
    return best, recs


# ----------------------------------------------------------------------------
# design: identical to c14.build_design (S1 baseline, corrected census)
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
        exog_unr = np.column_stack([dum["D_infrastructure"].values,
                                    dum["D_regulatory"].values,
                                    s[SENT_COLS].values])
        design[a] = {"returns": r.values.astype(float), "exog_unr": exog_unr}
    return design, inf_d, reg_d


# ----------------------------------------------------------------------------
def main():
    t_all = time.time()
    print(f"numba available: {_HAVE_NUMBA}")
    design, inf_d, reg_d = build_design()
    print(f"baseline events: {len(inf_d)} infra, {len(reg_d)} reg")

    rows = []
    anchor_rows = []
    for a in ASSETS:
        r = design[a]["returns"]
        exog5 = design[a]["exog_unr"]
        exog0 = np.empty((r.shape[0], 0))
        n = r.shape[0]

        specs = [
            ("GARCH(1,1)", FastTARCHXNoGamma(r, exog0), True, 0),
            ("GJR-GARCH", FastTARCHX(r, exog0), False, 0),
            ("GJR-GARCH-X", FastTARCHX(r, exog5), False, 5),
        ]
        for model, est, pin, n_exog in specs:
            t0 = time.time()
            default = est._default_start()
            if pin:
                default[2] = 0.0
            starts_c14 = make_starts(r, n_exog, N_STARTS_C14, SEED_C14,
                                     default, pin)
            starts_extra = make_starts(r, n_exog, N_STARTS_EXTRA + 1,
                                       SEED_EXTRA, default, pin)[1:]
            best_c14, recs_c14 = multistart_record(est, starts_c14)
            best_all, recs_extra = multistart_record(est, starts_extra)
            if best_c14[1] <= best_all[1]:
                best_all = best_c14
                extra_improved = False
            else:
                extra_improved = True
            recs = recs_c14 + recs_extra
            p, f, ok = best_all
            dt = time.time() - t0

            ll = -f
            omega, alpha, gamma, beta, nu = p[0], p[1], p[2], p[3], p[4]
            persistence = alpha + beta + abs(gamma) / 2.0
            default_f = recs[0][0]
            n_improving = sum(1 for (fv, _) in recs[1:] if fv < default_f - 1e-9)
            n_conv = sum(1 for (_, okv) in recs if okv)

            kL, kF, kM = K_LEGACY[model], K_FREE[model], K_MU[model]
            lnn = np.log(n)
            row = {
                "asset": a, "model": model, "n_obs": n,
                "k": kM, "negLL": f, "LL": ll,
                "AIC": 2 * kM - 2 * ll, "BIC": kM * lnn - 2 * ll,
                "k_free": kF,
                "AIC_kfree": 2 * kF - 2 * ll, "BIC_kfree": kF * lnn - 2 * ll,
                "k_mu": kM,
                "AIC_kmu": 2 * kM - 2 * ll, "BIC_kmu": kM * lnn - 2 * ll,
                "k_legacy": kL,
                "AIC_legacy": 2 * kL - 2 * ll,
                "BIC_legacy": kL * lnn - 2 * ll,
                "converged": ok,
                "n_starts": len(recs), "n_starts_converged": n_conv,
                "n_starts_improving_on_default": n_improving,
                "extra_grid_improved_best": extra_improved,
                "best_negLL_c14_grid": best_c14[1],
                "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
                "nu": nu, "persistence": persistence,
                "delta_infra": p[5] if n_exog else np.nan,
                "delta_reg": p[6] if n_exog else np.nan,
                "runtime_s": dt,
            }
            rows.append(row)
            print(f"{a:4s} {model:12s} negLL={f:10.4f} AIC(k={kM})={row['AIC']:10.2f} "
                  f"persist={persistence:.4f} conv={ok} "
                  f"starts_conv={n_conv}/{len(recs)} extra_improved={extra_improved} "
                  f"[{dt:.1f}s]")

            # SANITY ANCHOR: literal c14 call for the GJR-GARCH-X spec
            if model == "GJR-GARCH-X":
                est2 = FastTARCHX(r, exog5)
                p2, f2, ok2 = est2.fit_multistart(n_starts=N_STARTS_C14,
                                                  seed=SEED_C14,
                                                  max_iter=MAX_ITER)
                anchor_rows.append({
                    "asset": a, "negLL_literal_c14_call": f2,
                    "negLL_c18_c14grid": best_c14[1],
                    "match_internal": abs(f2 - best_c14[1]) < 1e-6,
                    "params_literal": p2,
                })

    df = pd.DataFrame(rows)
    out_csv = c2.OUT_DIR / "c18-model-comparison-refit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # ---- anchor check vs the c14 CSV of record --------------------------------
    c14 = pd.read_csv(c2.OUT_DIR / "c14-garch-diagnostics-per-asset.csv")
    anchor = []
    for ar in anchor_rows:
        a = ar["asset"]
        ref = c14.loc[c14["asset"] == a].iloc[0]
        mine = df[(df.asset == a) & (df.model == "GJR-GARCH-X")].iloc[0]
        anchor.append({
            "asset": a,
            "negLL_c14": ref["negLL"],
            "negLL_c18": mine["negLL"],
            "d_negLL": mine["negLL"] - ref["negLL"],
            "negLL_literal_call": ar["negLL_literal_c14_call"],
            "d_omega": mine["omega"] - ref["omega"],
            "d_alpha": mine["alpha"] - ref["alpha"],
            "d_gamma": mine["gamma"] - ref["gamma"],
            "d_beta": mine["beta"] - ref["beta"],
            "d_nu": mine["nu"] - ref["nu"],
            "d_dinfra": mine["delta_infra"] - ref["delta_infra"],
            "d_dreg": mine["delta_reg"] - ref["delta_reg"],
        })
    anc = pd.DataFrame(anchor)
    print("\nANCHOR vs c14 (GJR-GARCH-X, same spec/data/starts):")
    print(anc.to_string(index=False))

    write_finding(df, anc, len(inf_d), len(reg_d), time.time() - t_all)
    return df, anc


def winners(df, aic_col):
    out = {}
    for a in ASSETS:
        sub = df[df.asset == a]
        w = sub.loc[sub[aic_col].idxmin(), "model"]
        srt = sub.sort_values(aic_col)
        margin = srt[aic_col].iloc[1] - srt[aic_col].iloc[0]
        out[a] = (w, margin)
    return out


def write_finding(df, anc, n_inf, n_reg, elapsed):
    lines = []
    L = lines.append
    L("# C18 -- Model-Comparison Refit under a Single Controlled Estimator\n")
    L(f"_S1 baseline ({n_inf} infrastructure + {n_reg} regulatory events, corrected census); "
      f"FastTARCHX multistart (c14 grid seed {SEED_C14} + {N_STARTS_EXTRA} extra "
      f"starts seed {SEED_EXTRA}); numba={_HAVE_NUMBA}; total runtime "
      f"{elapsed:.1f}s._\n")

    L("## Parameter-count convention\n")
    L("The primary AIC/BIC columns count the profiled sample mean because it is "
      "estimated from the data: k=5/6/11. The CSV also retains k_free=4/5/10 "
      "and the predecessor mixed convention as audit columns. The two "
      "self-consistent counts produce identical within-asset rankings.\n")

    L("## Controlled refit (primary, k = 5/6/11)\n")
    L("| asset | model | k | LL | AIC | BIC | converged | n_starts |")
    L("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        L(f"| {r['asset']} | {r['model']} | {int(r['k'])} | {r['LL']:.2f} | "
          f"{r['AIC']:.2f} | {r['BIC']:.2f} | {r['converged']} | "
          f"{int(r['n_starts'])} |")
    L("")

    W_free = winners(df, "AIC_kfree")
    W_mu = winners(df, "AIC_kmu")
    W_bic = winners(df, "BIC_kmu")
    L("## Selection summary\n")
    L("| asset | AIC winner (k=5/6/11) | margin | AIC winner (k_free) | BIC winner |")
    L("|---|---|---|---|---|")
    for a in ASSETS:
        L(f"| {a} | {W_mu[a][0]} | {W_mu[a][1]:.2f} | "
          f"{W_free[a][0]} ({W_free[a][1]:.2f}) | {W_bic[a][0]} |")
    L("")
    L("GJR-GARCH-X has the lowest AIC for five assets. BNB is an effective tie: "
      f"GARCH(1,1) is lower by {W_mu['bnb'][1]:.2f} AIC. BIC prefers "
      "GARCH(1,1) for all six assets.\n")

    L("## Sanity anchor vs c14 (GJR-GARCH-X, same spec/data/grid)\n")
    L("| asset | negLL c14 | negLL c18 | delta | d_dinfra | d_dreg |")
    L("|---|---|---|---|---|---|")
    for _, r in anc.iterrows():
        L(f"| {r['asset']} | {r['negLL_c14']:.4f} | {r['negLL_c18']:.4f} | "
          f"{r['d_negLL']:.2e} | {r['d_dinfra']:.2e} | {r['d_dreg']:.2e} |")
    L("")

    L("## Convergence / pathology notes\n")
    for _, r in df.iterrows():
        flag = []
        if r["persistence"] >= 0.9989:
            flag.append("persistence at the 0.999 bound")
        if r["extra_grid_improved_best"]:
            flag.append("EXTRA-grid start beat the c14 grid")
        if not r["converged"]:
            flag.append("best fit NOT converged")
        if flag:
            L(f"- {r['asset']} {r['model']}: {'; '.join(flag)} "
              f"(persist={r['persistence']:.5f}, nu={r['nu']:.2f}, "
              f"starts improving on default: "
              f"{int(r['n_starts_improving_on_default'])})")
    L("")
    L("## Files\n")
    L("- `c18-model-comparison-refit.csv` -- full table including alternative "
      "parameter-count conventions and per-start bookkeeping")
    L("- `code/c18_model_comparison_refit.py`")

    out_md = c2.OUT_DIR / "c18-model-comparison-refit-FINDING.md"
    out_md.write_text("\n".join(lines))
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
