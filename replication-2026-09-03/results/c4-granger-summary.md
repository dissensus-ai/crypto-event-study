# C4 — Granger Causality (Sentiment vs Volatility): Summary

Pairwise Granger causality tests at lags 1-10 between decomposed GDELT sentiment and absolute daily log returns, per asset. Weekly GDELT data are forward-filled to daily, repeating weekly observations and changing the effective information content; these daily-lag results should not be interpreted as tests of weekly lead-lag effects. Reports the minimum p-value across lags and the AIC/BIC-optimal lag.

## Direction Summary (min-p across lags 1-10)

| Asset | Sentiment | F: sent→ret | p: sent→ret | F: ret→sent | p: ret→sent | Pattern |
|---|---|---|---|---|---|---|
| BTC | S_reg_decomposed | 0.46 (L10) | **0.4964** | 1.19 (L10) | **0.2907** | Neither (independent) |
| BTC | S_infra_decomposed | 1.63 (L10) | **0.2013** | 0.82 (L8) | **0.4385** | Neither (independent) |
| BTC | S_gdelt_normalized | 1.28 (L10) | **0.2776** | 1.00 (L10) | **0.4040** | Neither (independent) |
| ETH | S_reg_decomposed | 0.92 (L10) | **0.5145** | 1.29 (L9) | **0.2386** | Neither (independent) |
| ETH | S_infra_decomposed | 0.92 (L10) | **0.3993** | 1.06 (L9) | **0.3924** | Neither (independent) |
| ETH | S_gdelt_normalized | 0.92 (L10) | **0.5128** | 1.36 (L10) | **0.1995** | Neither (independent) |
| XRP | S_reg_decomposed | 1.28 (L10) | **0.2381** | 3.25 (L10) | **0.0716** | Neither (independent) |
| XRP | S_infra_decomposed | 1.63 (L10) | **0.1794** | 5.94 (L10) | **0.0148** | Returns LEAD |
| XRP | S_gdelt_normalized | 1.57 (L10) | **0.1102** | 5.18 (L10) | **0.0230** | Returns LEAD |
| BNB | S_reg_decomposed | 2.65 (L10) | **0.1034** | 2.92 (L10) | **0.0012** | Returns LEAD |
| BNB | S_infra_decomposed | 0.17 (L10) | **0.6836** | 1.50 (L10) | **0.1343** | Neither (independent) |
| BNB | S_gdelt_normalized | 2.12 (L10) | **0.1459** | 2.08 (L10) | **0.0232** | Returns LEAD |
| LTC | S_reg_decomposed | 0.68 (L10) | **0.7485** | 1.04 (L10) | **0.4075** | Neither (independent) |
| LTC | S_infra_decomposed | 1.65 (L10) | **0.1930** | 0.70 (L9) | **0.4036** | Neither (independent) |
| LTC | S_gdelt_normalized | 0.74 (L10) | **0.4795** | 1.29 (L10) | **0.2274** | Neither (independent) |
| ADA | S_reg_decomposed | 1.27 (L10) | **0.2400** | 2.08 (L10) | **0.1254** | Neither (independent) |
| ADA | S_infra_decomposed | 3.59 (L10) | **0.0582** | 0.06 (L8) | **0.7999** | Neither (independent) |
| ADA | S_gdelt_normalized | 0.46 (L10) | **0.4982** | 0.77 (L10) | **0.4640** | Neither (independent) |

## Pattern Counts (across 6 assets × 3 sentiment series)

- **Neither (independent)**: 14 pairs
- **Returns LEAD**: 4 pairs

## Interpretation

**Regulatory sentiment** (n=6 assets): 0 lead-only, 1 lag-only, 0 bidirectional, 5 neither.

**Infrastructure sentiment** (n=6 assets): 0 lead-only, 1 lag-only, 0 bidirectional, 5 neither.

The H2 claim of 'sentiment provides incremental explanatory power' is direction-agnostic. These tests disambiguate. Where infrastructure sentiment lags returns, the sentiment proxy is best interpreted as concurrent or post-hoc commentary rather than a leading indicator. Where regulatory sentiment leads returns, leakage and anticipation channels are consistent with the result. Section 4.4.5 in the revised manuscript reports these directions honestly.

## Files

- Full per-asset / per-direction table: `c4-granger-results.csv`
