# C13 -- Rung-4 design-effect sensitivity diagnostic

_Per-asset fitted-null t-copula bootstrap: requested B=2000, retained 2000, excluded 0; numba=False; runtime 12.8 min._

## Inferential status

This exercise is a heuristic diagnostic, not the paper's test of record and not a source of a replacement p-value. It asks how the analytical rung changes when its dependence input and tail reference are varied. No convention below is asserted to be the exact finite-sample law of six estimated GARCH-X contrasts.
The mapping df_eff=(N-1)/DEFF is an effective-df sensitivity convention. A design effect does not by itself prove that the resulting statistic is Student-t with that many degrees of freedom. Likewise, rho_d_bar is estimated from retained draws under the fitted-null c9 simulator, so it is more closely aligned with the averaged estimator than raw-return correlation but remains model-, screen-, seed-, and Monte-Carlo-dependent.

## Inputs and retained-draw diagnostics

- per-asset d_i = [0.6061, 1.7291, 1.1293, 0.8727, 1.9801, 2.432]
- observed mean_d = 1.4582; six-contrast dispersion SE = 0.2870; variance-increment ratio = 3.488x
- rho_return = 0.6882; rho_resid = 0.7046; fitted-null rho_d_bar = 0.3108
- retained-draw covariance ratio DEFF_cov = 1.994; equicorrelation formula at rho_d_bar = 2.554
- draws invoking the one-rescue rule = 127; asset-level rescue attempts/successes = 131/131

DEFF_cov is an algebraic variance ratio computed from the estimated covariance matrix of retained bootstrap draws. Calling it a covariance ratio avoids implying an exact population design effect.

### Correlation matrix of fitted-null per-asset difference estimates

| | btc | eth | xrp | bnb | ltc | ada |
|---|---|---|---|---|---|---|
| btc | 1.000 | 0.281 | 0.131 | 0.174 | 0.234 | 0.213 |
| eth | 0.281 | 1.000 | 0.280 | 0.423 | 0.475 | 0.459 |
| xrp | 0.131 | 0.280 | 1.000 | 0.185 | 0.308 | 0.361 |
| bnb | 0.174 | 0.423 | 0.185 | 1.000 | 0.354 | 0.343 |
| ltc | 0.234 | 0.475 | 0.308 | 0.354 | 1.000 | 0.439 |
| ada | 0.213 | 0.459 | 0.361 | 0.343 | 0.439 | 1.000 |

## Analytical reference sensitivity

All tail areas are one-sided for the pre-specified infrastructure-greater-than-regulatory direction. Each row applies the same formulas to a different dependence proxy; the columns then vary the tail reference.

| dependence input | rho | DEFF | N_eff | df_eff convention | t (dispersion) | p: t(df_eff) | p: t(N-1) | p: normal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho_return (c6 legacy input) | 0.688 | 4.44 | 1.35 | 1.13 | 2.41 | 0.1137 | 0.0304 | 0.0080 |
| rho_resid (standardized-residual sensitivity) | 0.705 | 4.52 | 1.33 | 1.11 | 2.39 | 0.1164 | 0.0312 | 0.0084 |
| rho_d_bar (fitted-null estimator-difference sensitivity) | 0.311 | 2.55 | 2.35 | 1.96 | 3.18 | 0.0444 | 0.0123 | 0.0007 |

Across the three dependence inputs for the dispersion statistic under the t(df_eff) convention, the descriptive p-value span is 0.044-0.116. For the dispersion statistic alone, changing rho_return (0.688) to fitted-null rho_d_bar (0.311) changes DEFF from 4.44 to 2.55 and the corresponding tail area from 0.114 to 0.044. This span is a sensitivity range, not a confidence interval and not a multiple-testing adjustment.

## Conditional rejection-rate diagnostic using c10 panels

The saved c10 panel statistic is recovered by inverting its t(N-1) tail area and is then compared with alternative critical values. The resulting rates are conditional on the c10 fitted DGP, retained panels, refit screen, and simulation settings; they are not estimates of unconditional size under the unknown true DGP.

| reference convention | critical value at .05 | critical value at .10 | conditional rejection rate at .05 | conditional rejection rate at .10 | retained panels |
|---|---:|---:|---:|---:|---:|
| normal z (size study's z=1.645) | 1.645 | 1.282 | 0.540 | 0.674 | 1000 |
| t(N-1=5) [c10 saved rule] | 2.015 | 1.476 | 0.370 | 0.618 | 1000 |
| t(df_eff=1.13) [sensitivity, rho_return] | 5.241 | 2.736 | 0.009 | 0.170 | 1000 |
| t(df_eff=1.96) [sensitivity, rho_d_bar] | 2.963 | 1.904 | 0.127 | 0.422 | 1000 |

Large movement across these rows demonstrates reference sensitivity. It does not identify one row as calibrated outside the fitted-DGP experiment, and it does not independently validate the fixed-path or recursive bootstrap.

## Interpretation

The defensible conclusion is narrow: the closed-form rung is highly sensitive to choices that are difficult to justify with only six fitted asset contrasts. Raw-return correlation is a legacy proxy; fitted-null corr(d_i,d_j) is more target-aligned but simulation-conditional. The effective-df tail is a heavier-tailed reference experiment, not a derived sampling theorem. Accordingly, rung 4 should remain a diagnostic range and should not be used to select significance or to certify the paper's bootstrap inference.

This diagnostic neither proves that the raw-return input is invalid nor that the estimator-difference input is true. It also does not show that one innovation law is correct for the unknown DGP. Those questions are addressed, only conditionally, by the separately reported fitted-DGP calibration and scheme-sensitivity analyses.

## Files

- `c13-rung4-recompute-results.csv` -- analytical sensitivity table and conditional c10 rejection-rate table
- `c13-rung4-perasset-draws.npz` -- retained per-asset difference draws and their estimated correlation/covariance matrices
- `code/c13_rung4_recompute.py` -- diagnostic generator using the c7/c9 fitted-null engine and saved c10 draws
