# C18 -- Model-Comparison Refit under a Single Controlled Estimator

_S1 baseline (26 infrastructure + 24 regulatory events, corrected census); FastTARCHX multistart (c14 grid seed 12345 + 6 extra starts seed 20260806); numba=False; total runtime 54.9s._

## Parameter-count convention

The primary AIC/BIC columns count the profiled sample mean because it is estimated from the data: k=5/6/11. The CSV also retains k_free=4/5/10 and the predecessor mixed convention as audit columns. The two self-consistent counts produce identical within-asset rankings.

## Controlled refit (primary, k = 5/6/11)

| asset | model | k | LL | AIC | BIC | converged | n_starts |
|---|---|---|---|---|---|---|---|
| btc | GARCH(1,1) | 5 | -5927.31 | 11864.63 | 11893.62 | True | 12 |
| btc | GJR-GARCH | 6 | -5927.31 | 11866.63 | 11901.41 | True | 12 |
| btc | GJR-GARCH-X | 11 | -5920.49 | 11862.98 | 11926.75 | True | 12 |
| eth | GARCH(1,1) | 5 | -6643.31 | 13296.63 | 13325.62 | True | 12 |
| eth | GJR-GARCH | 6 | -6643.31 | 13298.63 | 13333.41 | True | 12 |
| eth | GJR-GARCH-X | 11 | -6631.93 | 13285.85 | 13349.62 | True | 12 |
| xrp | GARCH(1,1) | 5 | -6630.67 | 13271.34 | 13300.32 | True | 12 |
| xrp | GJR-GARCH | 6 | -6630.48 | 13272.96 | 13307.75 | True | 12 |
| xrp | GJR-GARCH-X | 11 | -6621.24 | 13264.49 | 13328.26 | True | 12 |
| bnb | GARCH(1,1) | 5 | -5667.80 | 11345.60 | 11374.06 | True | 12 |
| bnb | GJR-GARCH | 6 | -5667.68 | 11347.35 | 11381.51 | True | 12 |
| bnb | GJR-GARCH-X | 11 | -5662.13 | 11346.26 | 11408.87 | True | 12 |
| ltc | GARCH(1,1) | 5 | -6858.25 | 13726.50 | 13755.49 | True | 12 |
| ltc | GJR-GARCH | 6 | -6855.96 | 13723.93 | 13758.71 | True | 12 |
| ltc | GJR-GARCH-X | 11 | -6849.99 | 13721.97 | 13785.74 | True | 12 |
| ada | GARCH(1,1) | 5 | -7014.89 | 14039.79 | 14068.77 | True | 12 |
| ada | GJR-GARCH | 6 | -7014.77 | 14041.54 | 14076.33 | True | 12 |
| ada | GJR-GARCH-X | 11 | -7007.86 | 14037.72 | 14101.49 | True | 12 |

## Selection summary

| asset | AIC winner (k=5/6/11) | margin | AIC winner (k_free) | BIC winner |
|---|---|---|---|---|
| btc | GJR-GARCH-X | 1.65 | GJR-GARCH-X (1.65) | GARCH(1,1) |
| eth | GJR-GARCH-X | 10.78 | GJR-GARCH-X (10.78) | GARCH(1,1) |
| xrp | GJR-GARCH-X | 6.85 | GJR-GARCH-X (6.85) | GARCH(1,1) |
| bnb | GARCH(1,1) | 0.66 | GARCH(1,1) (0.66) | GARCH(1,1) |
| ltc | GJR-GARCH-X | 1.96 | GJR-GARCH-X (1.96) | GARCH(1,1) |
| ada | GJR-GARCH-X | 2.07 | GJR-GARCH-X (2.07) | GARCH(1,1) |

GJR-GARCH-X has the lowest AIC for five assets. BNB is an effective tie: GARCH(1,1) is lower by 0.66 AIC. BIC prefers GARCH(1,1) for all six assets.

## Sanity anchor vs c14 (GJR-GARCH-X, same spec/data/grid)

| asset | negLL c14 | negLL c18 | delta | d_dinfra | d_dreg |
|---|---|---|---|---|---|
| btc | 5920.4886 | 5920.4886 | -5.83e-06 | -3.67e-04 | -1.02e-04 |
| eth | 6631.9267 | 6631.9267 | 0.00e+00 | 0.00e+00 | 5.55e-17 |
| xrp | 6621.2441 | 6621.2441 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| bnb | 5662.1303 | 5662.1303 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| ltc | 6849.9868 | 6849.9868 | -1.53e-09 | 1.30e-04 | 4.30e-05 |
| ada | 7007.8614 | 7007.8614 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

## Convergence / pathology notes

- btc GARCH(1,1): persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.24, starts improving on default: 9)
- btc GJR-GARCH: persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.24, starts improving on default: 1)
- btc GJR-GARCH-X: persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.13, starts improving on default: 3)
- eth GARCH(1,1): persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.67, starts improving on default: 4)
- eth GJR-GARCH: persistence at the 0.999 bound (persist=0.99900, nu=3.67, starts improving on default: 3)
- eth GJR-GARCH-X: persistence at the 0.999 bound (persist=0.99900, nu=3.77, starts improving on default: 0)
- xrp GARCH(1,1): persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.03, starts improving on default: 10)
- xrp GJR-GARCH: persistence at the 0.999 bound (persist=0.99900, nu=3.11, starts improving on default: 5)
- xrp GJR-GARCH-X: persistence at the 0.999 bound (persist=0.99900, nu=3.16, starts improving on default: 0)
- bnb GARCH(1,1): persistence at the 0.999 bound (persist=0.99900, nu=4.03, starts improving on default: 2)
- bnb GJR-GARCH: persistence at the 0.999 bound (persist=0.99900, nu=4.13, starts improving on default: 2)
- bnb GJR-GARCH-X: persistence at the 0.999 bound (persist=0.99900, nu=4.22, starts improving on default: 0)
- ltc GJR-GARCH: persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.92, starts improving on default: 3)
- ltc GJR-GARCH-X: persistence at the 0.999 bound; EXTRA-grid start beat the c14 grid (persist=0.99900, nu=3.94, starts improving on default: 1)
- ada GJR-GARCH: EXTRA-grid start beat the c14 grid (persist=0.97088, nu=4.46, starts improving on default: 4)

## Files

- `c18-model-comparison-refit.csv` -- full table including alternative parameter-count conventions and per-start bookkeeping
- `code/c18_model_comparison_refit.py`
