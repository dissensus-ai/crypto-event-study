# C10 -- Monte-Carlo Size-Distortion Study of the Inference Ladder

_N=1000 true-null panels (of 1000 requested); bootstrap critical values calibrated from B_ref=4000; numba=False; runtime 47.9 min._

## The question

How often do the inference rules reject under the fitted fixed-design null? This is an internal calibration diagnostic. It directly measures over-rejection of the naive rule in this DGP, but it does not independently validate the plug-in bootstrap because panels and critical values share the same fitted generator.

## DGP (true null, fitted to the data)

- Per-asset **GJR-GARCH-X** variance dynamics estimated under the **null-imposed** combined-dummy spec (D_event = D_infra + D_reg), so delta_infra = delta_reg **by construction** -- there is genuinely no differential event effect in the truth.
- Cross-asset dependence at the **standardised-residual correlation** rho_resid = 0.705 (return-correlation rho_return = 0.688), via the Cholesky of R_z.
- **Student-t innovations** at each asset's fitted nu = [3.158;3.696;3.158;4.218;3.959;4.559] (median nu_c = 3.828), unit variance, with joint tail dependence via a true multivariate-t copula (shared chi-square mixing). The fit seed and schema-v2 DGP, design, refitter, simulator, and analysis fingerprints are asserted exactly against c9 before any Monte-Carlo draws.
- The **actual 26 infra / 24 reg** event-label structure and the real event-window dummies are reused unchanged for every panel; only returns are re-simulated. (Point estimate on the real data: d_bar = 1.458, 3.49x -- the SIM imposes the null, this is shown only to anchor the DGP.)

## The size table (empirical rejection rate under a true null)

| method | size @ alpha=0.05 | size @ alpha=0.10 |
|---|---|---|
| (i) NAIVE iid t-test | **0.351** +/- 0.015 | **0.526** +/- 0.016 |
| (ii) design-effect corrected | 0.370 +/- 0.015 | 0.618 +/- 0.015 |
| (iii) Gaussian-copula bootstrap | 0.288 +/- 0.014 | 0.387 +/- 0.015 |
| (iv) Student-t-copula bootstrap | 0.046 +/- 0.007 | 0.099 +/- 0.009 |

_(+/- = Monte-Carlo binomial standard error.) Nominal size is 0.05 and 0.10._

## Internal self-consistency check

The Student-t-copula rule is calibrated from the same fixed DGP used to draw the panels, so near-nominal rejection is an expected implementation check, not independent validation. It comes out **0.046 @0.05** and **0.099 @0.10** (targets 0.05/0.10). Relative to the reported binomial Monte Carlo standard errors, the deviations are **-0.60 SE** and **-0.11 SE**, respectively.

## Verdict on the naive over-rejection

- The **naive iid t-test rejects a true null 35.1% of the time at the 5% level (7.0x nominal) and 52.6% at the 10% level (5.3x nominal).** This is severe size distortion: a 'significant' headline from this rule is, under the fitted null, a false positive a large fraction of the time. It is the direct, simulated counterpart of the pseudoreplication diagnosis -- the six assets are not six independent draws.
- The **design-effect t(5) rule** rejects at 0.370/0.618, versus 0.351/0.526 for the naive rule. It therefore does not reduce over-rejection in this run. The rule uses raw-return correlation as a proxy and does not reproduce the fitted GARCH/heavy-tail structure.
- The **Gaussian-copula bootstrap** lands at 0.288/0.387 and the **Student-t-copula bootstrap** at 0.046/0.099 -- under this fitted DGP. The load-bearing result is the severe over-rejection of the naive and Gaussian rules; the t-copula result is internally calibrated.

## Method notes

- (i) NAIVE: Welch t-test on the 6 per-asset delta_infra vs 6 delta_reg as iid (the headline rule that produced t=4.768/p=0.0008).
- (ii) DESIGN-EFFECT: paired-difference SE inflated by sqrt(1+(N-1)*rho_bar) on the cross-asset RETURN correlation, t on N-1 df (c6's rule).
- (iii)/(iv) BOOTSTRAP: ONE-SIDED upper-tail test (H1: infra>reg) -- reject if d_bar exceeds the (1-alpha) empirical quantile of the respective null sampling distribution of d_bar (Gaussian copula = c7; Student-t copula = c9). This is the critical-value analogue of the c7/c9 upper-tail decision; at finite B it is not literally identical to evaluating the add-one p-value panel by panel, and a boundary decision can differ by one Monte-Carlo draw. Critical values are calibrated once from B_ref reference draws under the same fixed DGP. This tests self-consistency; a fully nested re-estimation would be required for independent plug-in validation. One-sided critical d_bar: Gaussian 5%=1.6234/10%=1.4559, t-copula 5%=2.5173/10%=2.1788 (null centres 0.8981 / 1.2609).

## Fragility / honest caveats

- **Convergence:** 0.0% of the 1000 outer panels remained excluded after the shared deterministic rescue (a per-asset refit failed to converge or hit a degenerate |delta|>50 optimum). Reference-draw drop rates: Gaussian 0.1%, t-copula 0.0%. All under the 10% reliability threshold (reliable).
- **Per-asset model SEs are audit-only, not on the decision path.** Neither rung (i) (Welch on the 6 vs 6 coefficients) nor rung (ii) (dispersion of the 6 paired differences, c6's rule) uses the per-asset model SE, so an SE failure never drops a panel or flips a decision. The numerical-Hessian SEs are computed only for reporting; their sub-block was positive-definite for 99.2% of accepted panels. Panel acceptance is therefore identical to the reference-draw rule (convergence + |delta|<=50), and refit screen matches the reference-draw rule.
- **The bootstrap rungs are calibrated, not fully nested.** Rungs (iii)/(iv) measure self-consistency under one shared fitted fixed-path DGP. A fully nested design that re-estimates the null DGP and critical values for every panel would be required for independent plug-in validation.
- **Copula df** for the t-copula tail-mixing is the median fitted nu (one shared dependence parameter), with fitted per-asset Student-t margins. The Gaussian comparator retains the same fitted R_z linear-correlation input but uses Gaussian margins and has no shared-mixing tail dependence. Their contrast therefore changes the full joint innovation law---both margins and tail dependence---rather than isolating a margins-only effect; neither law is guaranteed correct under the unknown true DGP.
- **Monte-Carlo error**: sizes carry binomial SEs of ~0.015-0.016; differences within ~2 SE of each other or of nominal are not separable at this N.

## Files

- `c10-size-study-results.csv` -- method-by-level size table
- `c10-size-study-metadata.csv` -- DGP, screening, and calibration metadata
- `c10-size-study-draws.npz` -- full seed-indexed panel coefficients/SEs, per-panel p-values (iid/deff), reference null draws, masks, critical values, and exact DGP/calendar provenance
- `code/c10_size_study.py` (reuses `tarch_x_fast.py`, `c7_ccc_garchx_bootstrap.py`, `c9_tcopula_bootstrap.py`)
