# C2 — Relaxed-Threshold Sensitivity: Summary

**Headline question.** How does the C2 pipeline's mechanically reconstructed S1 multiplier change when the Stage-2 impact filter is applied mechanically, relaxed, or removed?

**Answer.** See the table below — the multiplier under each specification.

## Cross-Asset Multiplier by Specification

| Spec | N infra | N reg | δ̄ infra | δ̄ reg | Multiplier | Cohen's d | Welch t | p |
|---|---|---|---|---|---|---|---|---|
| S1_baseline | 26 | 24 | 2.0443 | 0.5863 | **3.487×** | 2.075 | 3.594 | 0.0059 |
| S2_relaxed | 69 | 46 | 1.4709 | 1.1699 | **1.257×** | 0.321 | 0.556 | 0.5926 |
| S3_nofilter | 82 | 53 | 0.2600 | 0.5345 | **0.487×** | -0.679 | -1.176 | 0.2876 |
| S4_strict | 42 | 36 | 3.9911 | 3.0096 | **1.326×** | 0.526 | 0.910 | 0.3843 |

## Interpretation

The mechanical specifications are descriptive inclusion-rule sensitivities.
Their Welch p-values treat the six per-asset coefficients as independent and
are not the paper's final inference. The attenuation relative to the curated
sample is a scope condition; it does not identify selection bias in upstream
manual curation.

- Relaxed-threshold multiplier: **1.257×** (p = 0.5926)
- No-filter multiplier:         **0.487×**
- Strict-threshold multiplier:  **1.326×**

## Files

- Per-asset model parameters: `c2-relaxed-threshold-results.csv`
- Cross-asset summary: `c2-summary-table.csv`
- Plot: `c2-multiplier-decay.png` / `.pdf`
