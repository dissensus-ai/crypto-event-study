# C10b -- DGP-perturbation sensitivity

_N=300 common-random-number panel positions per scenario; B_ref=2000; fit seed=12345; Monte Carlo seed=20260618._

This is a sensitivity study within specified fitted null DGPs, not independent validation under the unknown empirical DGP. S0 is asserted to be the exact seed-indexed prefix of the final c10 run; every scenario uses the same panel and reference seed positions.

| scenario | usable | naive iid | design-effect t(5) | Gaussian comparator | Student-t copula |
|---|---:|---:|---:|---:|---:|
| S0 baseline (fitted nu, matched) | 300 | 0.360 | 0.387 | 0.323 | 0.057 |
| S1 lighter tails nu=8 (matched) | 300 | 0.440 | 0.447 | 0.120 | 0.077 |
| S2 heavier tails nu=2.5 (matched) | 300 | 0.343 | 0.333 | 0.557 | 0.050 |
| S3 near-Gaussian t-copula (nu_c=200, fitted margins) | 300 | 0.360 | 0.350 | 0.297 | 0.067 |
| S4 MIS-SPECIFIED (simulate nu=2.5, calibrate at fitted nu) | 300 | 0.343 | 0.333 | 0.557 | 0.260 |

Entries are empirical rejection rates at nominal 5%; the CSV also contains the 10% rates and binomial Monte Carlo standard errors. The near-Gaussian row retains fitted Student-t margins and sets the Student-t copula df to 200; it is not a fully Gaussian innovation law. The deliberately misspecified final row separates simulation and calibration tails.

Files: `c10b-dgp-perturbation-results.csv` and `c10b-dgp-perturbation-draws.npz`.
