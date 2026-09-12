# C9 -- Student-t-copula CCC-GARCH-X fixed-path bootstrap

_B=2000; copula=true multivariate-t (shared chi-square mixing) with per-asset Student-t margins at the FITTED nu; runtime 97.2 min._

## Motivation

c7 and c8h are the cross-asset-robust significance tests for the infrastructure-vs-regulatory variance-coefficient asymmetry. Their CCC parametric bootstrap drew the standardised innovations as **Gaussian** (`rng.standard_normal`) even though each GJR-GARCH-X is fitted with **Student-t** errors, nu ~ 3.1-4.6. The Gaussian draw therefore does not match the fitted innovation margins or their joint tail behaviour. The c10 internal calibration shows material over-rejection for that comparator in the fitted design; the direction is not asserted as universal.

## The fix

Innovations are now drawn from a Student-t copula: latent MVN(0, R_z) divided by a shared chi-square (-> multivariate-t, joint tail dependence), mapped to uniforms, then to per-asset Student-t margins at the **fitted nu**, rescaled to unit variance by sqrt((nu-2)/nu). Same change in the null-imposed and unrestricted draws; one joint six-vector is generated for every union-calendar date and unavailable coordinates are discarded. Thus the five pre-BNB assets retain the marginal dependence implied by R_z. Everything else (B, null via combined dummy, refit, drop guards) uses the shared c7/c8h engine.

## Results: Gaussian comparator vs Student-t specification

| spec | multiplier | Gaussian p (1-sided) | **t-copula p (1-sided)** | t-copula absolute-statistic p | null SD Gauss -> t |
|---|---|---|---|---|---|
| baseline | 3.49x | 0.1130 | **0.3883** | 0.3893 | 0.4755 -> 0.8043 |
| crisis | 2.90x | 0.1381 | **0.3942** | 0.3987 | 0.5080 -> 0.9064 |
| full | 1.94x | 0.2404 | **0.3282** | 0.3917 | 0.8240 -> 1.4372 |

The retained machine label `crisis` denotes one asset-specific high-variance window from each asset's last 2021 conditional-variance break to its first break in 2022 or later. Four windows end around FTX; BNB and ADA end in July 2022, so this is a broad regime control rather than an FTX-specific control.

## Exact draw-retention and rescue telemetry

The full NPZ arrays retain all requested seed positions; compressed legacy arrays contain only usable values. Rescue counts below refer to the single deterministic six-start retry after a failed or degenerate initial refit.
| spec | t retained / requested | t excluded | rescue draws attempted / retained / failed | asset refits attempted / rescued | Gaussian retained / requested | Gaussian excluded |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2000 / 2000 | 0 | 127 / 127 / 0 | 131 / 131 | 1999 / 2000 | 1 |
| crisis | 1998 / 2000 | 2 | 162 / 160 / 2 | 167 / 165 | 1997 / 2000 | 3 |
| full | 1998 / 2000 | 2 | 459 / 457 / 2 | 513 / 511 | 2000 / 2000 | 0 |

## Direct comparison under matched seeds

The two innovation specifications are rerun with the same seeds and fixed-path design. A change in p has no predetermined validation direction; the relevant checks are the intended margins, dependence, and stated calibration results:
- **baseline**: p 0.1130 (Gaussian) -> 0.3883 (t-copula); null SD 0.4755 -> 0.8043.
- **crisis**: p 0.1381 (Gaussian) -> 0.3942 (t-copula); null SD 0.5080 -> 0.9064.
- **full**: p 0.2404 (Gaussian) -> 0.3282 (t-copula); null SD 0.8240 -> 1.4372.

**Interpretation.** The fitted Student-t margins are rescaled to unit variance. The table reports the realised direction and magnitude of each change for these three fitted specifications. Neither a wider-null heuristic nor a predetermined direction of the p-value is a general correctness test.
## Per-spec verdict

- **baseline** (3.49x, d_bar_obs=1.458): NOT significant at 10% -- directional only (p=0.3883).
- **crisis** (2.90x, d_bar_obs=1.296): NOT significant at 10% -- directional only (p=0.3942).
- **full** (1.94x, d_bar_obs=1.216): NOT significant at 10% -- directional only (p=0.3282).

## Honest headline

With the specified Student-t innovations the baseline result is **p=0.3883 -- NOT SIGNIFICANT** (>0.10). The fitted Student-t specification gives a higher p than the Gaussian comparator in this design. The point estimate (3.49x) is unchanged; the conditional fixed-path inference does not reject. The recursive sensitivity is reported separately.

Single-regime-control stability (legacy machine key `crisis`): the asset-specific high-variance-regime p is 0.3942, versus baseline p=0.3883; full-regime p=0.3282.

## Caveats

- The copula df is set to the median fitted nu (a single shared tail-dependence parameter), while the margins use each asset's fitted nu. C20 reports sensitivity to alternative shared-df values; this choice is a modelling assumption, not a guarantee of conservatism.
- R_z is calibrated on the six-asset complete-case residual window. Simulation uses the union calendar: a six-vector and one shared chi-square mixer are drawn per date, then unavailable asset coordinates are discarded. This uses the corresponding marginal of R_z on five-asset pre-BNB dates; it extrapolates the complete-case dependence estimate to that earlier overlap.
- Bootstrap refits use a single default start followed, when necessary, by a deterministic six-start rescue. Remaining degenerate refits are excluded and counted. The reported p-values use add-one Monte Carlo smoothing over retained refits, p=(hits+1)/(B_used+1); the full arrays and masks disclose the excluded seed positions.
- `USE_T_COPULA` toggles true-t-copula vs Gaussian-copula-with-t-margins; the headline above uses the true t-copula.

## Files

- `c9-tcopula-results.csv`, `c9-tcopula-draws-{baseline,crisis,full}.npz` (legacy compressed arrays plus full seed-indexed arrays, masks, and rescue telemetry)
- `code/c9_tcopula_bootstrap.py` (reuses `c7_ccc_garchx_bootstrap.py` + `c8h_break_controls_ccc_bootstrap.py` engines)