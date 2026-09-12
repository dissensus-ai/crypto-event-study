"""
c21: Event dummies in the MEAN equation (manuscript Section 7.3.2).

The paper's §7.3.2 robustness check demeans each return series by an OLS
regression on a constant and the same infrastructure/regulatory window dummies,
then refits the GJR-GARCH-X variance equation on those residuals -- so events
enter BOTH moments. Its reported numbers predate this rerun tree (the check was
run ad hoc and no script was preserved); this script reconstructs it so the
figures trace to committed code on the corrected census.

Reports: per-asset OLS mean coefficients on the two dummies, and the refitted
variance-equation event coefficients / cross-asset multiplier.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c7_ccc_garchx_bootstrap as c7
from c7_ccc_garchx_bootstrap import ASSETS
from tarch_x_fast import FastTARCHX

MAX_ITER = 2000
N_STARTS = 6
SEED = 12345


def main():
    t0 = time.time()
    design, inf_d, reg_d, ret_df = c7.build_design()
    rows = []
    for a in ASSETS:
        d = design[a]
        r = d["returns"].astype(float)
        exog = d["exog_unr"]                    # [D_infra, D_reg, S_gdelt, S_reg, S_infra]
        Dinf, Dreg = exog[:, 0], exog[:, 1]

        # --- OLS: r_t = c + b_infra*D_infra + b_reg*D_reg + e_t -------------
        X = np.column_stack([np.ones_like(r), Dinf, Dreg])
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        resid = r - X @ beta

        # --- refit the variance equation on the mean-filtered residuals ----
        est = FastTARCHX(resid, exog)
        p, f, ok = est.fit_multistart(n_starts=N_STARTS, seed=SEED, max_iter=MAX_ITER)
        rows.append({
            "asset": a, "converged": bool(ok),
            "ols_const": beta[0], "ols_b_infra": beta[1], "ols_b_reg": beta[2],
            "delta_infra": p[5], "delta_reg": p[6],
            "omega": p[0], "alpha": p[1], "gamma": p[2], "beta_g": p[3], "nu": p[4],
            "negLL": f,
        })
        print(f"  {a}: OLS b_infra={beta[1]:+.4f}% b_reg={beta[2]:+.4f}% | "
              f"delta_infra={p[5]:.4f} delta_reg={p[6]:.4f}")

    df = pd.DataFrame(rows)
    di, dr = df.delta_infra.mean(), df.delta_reg.mean()
    print("\n=== Events-in-mean (both moments) ===")
    print(f"  OLS mean coefficients: infra={df.ols_b_infra.mean():+.4f}%  reg={df.ols_b_reg.mean():+.4f}%")
    print(f"  variance: d_infra={di:.4f}  d_reg={dr:.4f}  multiplier={di/dr:.4f}x")
    print(f"  (constant-mean baseline of record: 2.0443 / 0.5861 / 3.4879x)")
    out = c7.c2.OUT_DIR if hasattr(c7, "c2") else HERE.parent
    import c2_relaxed_threshold_sensitivity as c2
    df.to_csv(c2.OUT_DIR / "c21-events-in-mean.csv", index=False)
    print(f"\nSaved {c2.OUT_DIR / 'c21-events-in-mean.csv'}")
    print(f"Total {time.time()-t0:.1f}s")
    print("C21_EVENTS_IN_MEAN_DONE")


if __name__ == "__main__":
    main()
