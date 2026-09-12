# C7 -- Gaussian-copula CCC-GARCH-X bootstrap (intermediate benchmark)

_Bootstrap B=500; numba=False; runtime 8.6 min._

## The question

Is the infrastructure-vs-regulatory variance-coefficient asymmetry (d_bar = mean_a(delta_infra - delta_reg) over 6 assets) significant under a model-based Gaussian-copula benchmark -- i.e. via a model-based bootstrap of the ACTUAL GJR-GARCH-X estimator, with cross-asset dependence calibrated to the STANDARDISED-residual correlation rather than the raw-return correlation c6 used as a post-hoc proxy? c6 bracketed p ~ 0.067-0.078 and conjectured it had OVER-penalised (residual corr assumed << return corr). This test checks that conjecture directly.

## Per-asset coefficients (baseline S1, 50 events)

| asset | delta_infra | delta_reg | diff |
|---|---|---|---|
| btc | 0.9067 | 0.3006 | 0.6061 |
| eth | 2.1102 | 0.3811 | 1.7291 |
| xrp | 2.8659 | 1.7366 | 1.1293 |
| bnb | 1.2313 | 0.3587 | 0.8727 |
| ltc | 2.3156 | 0.3355 | 1.9801 |
| ada | 2.8363 | 0.4043 | 2.4320 |

## Headline numbers

- **d_bar_obs** = 1.4582
- **multiplier** (mean infra / mean reg) = 3.488x
- **rho_bar(returns)** = 0.6882
- **rho_bar(standardised residuals)** = 0.7046  <- the crux. The pre-test hypothesis was that GARCH strips common volatility, so rho_resid would be MUCH LOWER than rho_return and c6's raw-return penalty would be too harsh. That did NOT hold: rho_resid (0.705) is actually slightly HIGHER than rho_return (0.688). One interpretation is that standardisation makes the shared component relatively more prominent, but this correlation comparison does not identify that mechanism. The fitted residual-dependence benchmark is nevertheless not weaker than the raw-return proxy used in c6.

## Significance

- **Parametric NULL-imposed (CCC MVN copula on R_z): one-sided p = 0.1038**, absolute-statistic p = 0.1038 (B used 500, dropped 0.0%). This redraws Gaussian standardised innovations against the fitted null variance paths with the fitted cross-asset correlation, refits unrestricted, and compares the observed d_bar to the null sampling distribution (including the estimator's finite-sample behaviour under this benchmark).
- Cross-sectional WILD (Rademacher, shared eta_t): one-sided p = 0.0020, absolute-statistic p = 0.0020 (B used 500, dropped 0.0%). **DO NOT TRUST this p as significance evidence.** Sign-flipping residuals barely perturbs a VARIANCE-equation coefficient because eps^2 is sign-invariant; the wild d_bar distribution is near-degenerate (sd ~0.1), so the tiny p is an artifact of under-dispersion, not power. It is reported only for completeness; it is the wrong instrument for this estimand.
- Unrestricted parametric CI for d_bar (bias-corrected basic bootstrap; the estimator is upward-biased by median 0.444 in this sample): 90% [0.2829, 1.7287], 95% [0.1682, 1.8546]; fraction of (bias-corrected) draws <= 0 = 0.0120 (raw percentile 95% [1.0619, 2.7483]; B used 500, dropped 0.0%)

## Comparison with c6 (p ~ 0.067-0.078)

The c7 benchmark (0.1038) is **weaker than c6's ~0.07** -- the asymmetry does not reach even 10% under this Gaussian-copula benchmark.

## Verdict within the c7 Gaussian-copula benchmark

Not significant at 10% under the parametric null-imposed CCC bootstrap (p=0.1038).

## Caveats (honest)

- This is an intermediate comparator, not the final inferential verdict. It uses a Gaussian copula (MVN on R_z) for cross-asset dependence of standardised innovations; the per-asset variance path is the fitted Student-t GJR model, but the cross-asset copula itself is Gaussian (tail dependence not modelled). With nu~3 marginals this could mildly understate joint tail co-movement; the direction of any resulting bias on the p-value is not obvious from c7 alone. C9 and c19 report the heavy-tail and propagation-scheme sensitivities used in the manuscript.
- **The wild bootstrap is NOT a usable anchor here** (contrary to the usual model-free role): Rademacher sign-flips leave eps^2 -- and hence the variance-equation event coefficients -- almost unchanged, collapsing the wild d_bar distribution (sd ~0.1) and producing an artificially tiny p. We flag this rather than hide it.
- The unrestricted GARCH event-coefficient estimator is upward-biased in this small sample (bootstrap median bias ~0.44); the null-imposed test handles this automatically (its null distribution is centred at the biased-under-equality location, not at 0), and the reported CI is the bias-corrected basic-bootstrap interval.
- Bootstrap refits use a single default start followed, on failure or a degenerate optimum, by exactly one deterministic six-start rescue; the observed fits use multistart. Refits that remain degenerate (SLSQP 'success' but |delta|>50, e.g. beta->0 / omega blown up) are rejected and counted as dropped.
- Convergence: dropped fractions above are the share of draws still unusable after that common rescue and are reported rather than silently replaced.

## Files

- `c7-ccc-bootstrap-results.csv` -- per-asset + summary
- `c7-bootstrap-draws.npz` -- raw d_bar draws (null/unr/wild)
- `code/c7_ccc_garchx_bootstrap.py`, `code/tarch_x_fast.py`