# C14 -- GARCH Squared-Residual Diagnostics (baseline GJR-GARCH-X)

_Baseline S1 spec (26 infra + 24 reg curated events); FastTARCHX multistart fit (same model as c6/c7); numba=False; runtime 11.8s._

## The question

Reviewer #C2 asks whether the GJR-GARCH-X variance equation fully absorbs conditional heteroskedasticity *before* the event coefficients are interpreted. If standardised residuals still carry ARCH, then unmodelled volatility clustering -- which is concentrated in the 2022-23 window where the infrastructure events (Terra/Luna, FTX, etc.) cluster -- could be absorbed by the D_infra dummy and mechanically inflate delta_infra and the ~3.49x curated multiplier. The test: Ljung-Box Q on z_t^2 at lags 5/10/20 plus Engle ARCH-LM, per asset.

## Fitted variance parameters (baseline)

| asset | n | omega | alpha | gamma | beta | nu | persist. | dInfra | dReg |
|---|---|---|---|---|---|---|---|---|---|
| btc | 2434 | 0.1223 | 0.0741 | -0.0000 | 0.9249 | 3.13 | 0.9990 | 0.9067 | 0.3006 |
| eth | 2434 | 0.1509 | 0.0697 | -0.0087 | 0.9250 | 3.77 | 0.9990 | 2.1102 | 0.3811 |
| xrp | 2434 | 1.3810 | 0.2207 | -0.0329 | 0.7619 | 3.16 | 0.9990 | 2.8659 | 1.7366 |
| bnb | 2191 | 0.2929 | 0.1433 | -0.0118 | 0.8498 | 4.22 | 0.9990 | 1.2313 | 0.3587 |
| ltc | 2434 | 0.6943 | 0.0985 | -0.0337 | 0.8837 | 3.94 | 0.9990 | 2.3156 | 0.3355 |
| ada | 2434 | 1.4427 | 0.1634 | -0.0266 | 0.7945 | 4.57 | 0.9713 | 2.8363 | 0.4043 |

_persist. = alpha + beta + |gamma|/2 (stationarity < 1)._

The numerical variance floor is not reached in any fitted series; the smallest pre-floor variance is 0.2373.

## Standardised-residual moments (whitening sanity)

| asset | mean(z) | mean(z^2) | excess kurt(z) |
|---|---|---|---|
| btc | 0.002 | 0.836 | 3.50 |
| eth | 0.003 | 0.920 | 2.58 |
| xrp | -0.008 | 0.891 | 5.84 |
| bnb | -0.016 | 0.957 | 2.67 |
| ltc | 0.001 | 0.947 | 2.23 |
| ada | -0.012 | 0.965 | 2.32 |

_mean(z)~0 and mean(z^2)~1 indicate the variance level is captured; residual excess kurtosis is expected (Student-t marginals) and is not an ARCH symptom._

## Ljung-Box Q on z_t^2 (residual-ARCH portmanteau)

| asset | Q(5) | p5 adj | Q(10) | p10 adj | Q(20) | p20 adj |
|---|---|---|---|---|---|---|
| btc | 6.66 | 0.0837 | 11.84 | 0.1587 | 24.21 | 0.1481 |
| eth | 1.27 | 0.7373 | 8.11 | 0.4225 | 16.51 | 0.5567 |
| xrp | 2.57 | 0.4633 | 4.95 | 0.7631 | 10.08 | 0.9292 |
| bnb | 3.70 | 0.2963 | 9.95 | 0.2685 | 19.18 | 0.3805 |
| ltc | 3.92 | 0.2707 | 7.08 | 0.5285 | 14.44 | 0.6997 |
| ada | 4.18 | 0.2425 | 8.74 | 0.3650 | 17.19 | 0.5098 |

_p..adj is a simple, more rejection-prone df = lag - 2 sensitivity that subtracts the ARCH and GARCH lag orders. It is not labelled as a general Li-Mak correction. Conventional df = lag p-values are in the CSV and give the same conclusion._

## Engle ARCH-LM on z_t (T*R^2 of z^2 on its lags)

| asset | LM(5) | p(5) | LM(10) | p(10) |
|---|---|---|---|---|
| btc | 6.74 | 0.2404 | 11.89 | 0.2924 |
| eth | 1.23 | 0.9421 | 9.58 | 0.4785 |
| xrp | 2.60 | 0.7619 | 5.08 | 0.8861 |
| bnb | 3.63 | 0.6040 | 9.70 | 0.4673 |
| ltc | 3.88 | 0.5666 | 7.10 | 0.7161 |
| ada | 4.26 | 0.5126 | 9.03 | 0.5295 |

## Significant residual-ARCH flags (p < 0.05)

- Ljung-Box(z^2), adjusted df: lag5 = 0/6, lag10 = 0/6, lag20 = 0/6 assets.
- Ljung-Box(z^2), naive df (for reference): lag5 = 0/6, lag10 = 0/6, lag20 = 0/6 assets.
- ARCH-LM: lag5 = 0/6, lag10 = 0/6 assets.
- Assets flagged by ANY adjusted test: none.

## Verdict

ADEQUATE. No asset shows significant residual ARCH at the 5% level on either Ljung-Box(z^2) convention (lags 5/10/20) or the Engle ARCH-LM test (lags 5/10). These diagnostics find no evidence of remaining ARCH at the tested lags. They do not prove that the variance model is complete or that the event coefficients are free of every possible dynamic misspecification.

## Caveats (honest)

- The portmanteau on squared standardised residuals is the standard McLeod-Li check for *remaining* ARCH after a GARCH fit; it does not test the level fit (mean(z^2)~1 covers that separately above).
- The df = lag - (p+q) = lag - 2 column is a deliberately more rejection-prone sensitivity, not a universal Li-Mak correction. Li and Mak (1994) motivate model-aware squared-residual diagnostics whose exact asymptotic adjustment can be nonstandard. The conventional df = lag column is also reported and yields the same conclusion. The result is therefore presented as a diagnostic rather than formal model certification.
- A clean portmanteau means no *linear* ARCH remains in z^2; it does not rule out higher-order nonlinearity or regime structure (those are handled separately by the Bai-Perron / persistence-break analyses in c3/c8).
- This diagnostic conditions on the fitted point estimates (multistart MLE); it is descriptive of model adequacy, not an inference test on the deltas (those are assessed by the c9 fixed-path bootstrap and the c19 recursive sensitivity).

## Files

- `c14-garch-diagnostics-per-asset.csv` -- full per-asset table
- `code/c14_garch_diagnostics.py`, `code/tarch_x_fast.py`, `code/c2_relaxed_threshold_sensitivity.py`