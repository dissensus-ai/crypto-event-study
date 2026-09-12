# C20 -- Paired copula-df sensitivity

_B=2000 requested per row; fit seed 12345; requested draw seeds 22345-24344; runtime 36.6 min._

## Pairing and provenance

All three rows use the same observed fit and the same ordered draw-seed array. The fitted-median row was rerun through c9's full diagnostic runner and matched the final c9 baseline exactly: full statistic vector including NaNs, usable/excluded masks, compressed vector, rescue telemetry, observed statistic, fitted marginal degrees of freedom, tail-hit counts, p-values, null mean, null SD, and audit hashes.

The value 5.9 is an alternative sensitivity value, not a rank-based estimate. No independently reproducible rank-based estimation artifact is included in this package. The value 8.0 is a second, lighter-tail copula-df sensitivity value. Per-asset fitted Student-t margins remain unchanged in every row; only the shared copula df changes.

## Results

| role | nu_c | used/requested | one-sided p | absolute-statistic p | null mean | null SD | paired usable | mean paired change vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fitted_median_baseline | 3.82768 | 2000/2000 | 0.3883 | 0.3893 | 1.2796 | 0.8043 | 2000 | +0.0000 |
| alternative_sensitivity | 5.9 | 2000/2000 | 0.3968 | 0.3983 | 1.2928 | 0.9408 | 2000 | +0.0131 |
| alternative_sensitivity | 8 | 2000/2000 | 0.4013 | 0.4018 | 1.2942 | 0.8054 | 2000 | +0.0146 |

Across these three tested values, the one-sided p-value range has width 0.0130 and the null-mean range has width 0.0146. Because exclusion masks can differ by copula df, rowwise p-values use each row's retained draws, whereas paired-change summaries use only the intersection of that row's and the fitted baseline's usable seed positions.

## Interpretation

This is a local, common-random-number sensitivity analysis. It does not estimate the copula df, validate the fixed-path bootstrap under the unknown data-generating process, or address the larger difference between the fixed-path and floored-recursive resampling schemes.

## Audit anchors

- fitted median nu_c: 3.8276848504612273
- c9 baseline NPZ SHA-256: cc191a5d9425434c979269ccdacfbdb6867fd8e6b9cbb8114b4b2d3865e8dd71
- c9 results CSV SHA-256: 9817c2c7414cbb304a6f08b29329e6454ec832568d231c6df19cd9a839eb044e
- design hash: f17b510ac2f8de93b42f24de232fb53d7b4ab720bf7188b4679e60ec56e5d274
- fixed-null DGP hash: 65d4781f5428f1e1ded2984186ac5041bfb8424f792e59d6d917865acfd64eee
- refitter hash: d7f819befd3a1335ea5a0da558cd031b20b0ee2263243c317711546302819356
- simulator hash: 925ce3b711aa0785e4356e75434cad1a33cc176580ba00839f622f2d9f06764a
- analysis-stack hash: 46c3941d3304dda42d135d04e10db3f4a9f20490f5ec17b69f76e2fab0cdc612
- c20 analysis hash: d5f3d7aa4a7dbe5928ebd03da5300f5305cd65b250421095494e3ee3424937de
- c20 sensitivity-laws hash: 47a8b9639ef3b2bffc3a7f77c8c6ffd212d6c66e88ffbee9f716ecf55ab6d552
- c20 source hash: 0a7febce0e27f6b1b988273da9ff2403654ba6cbc43ba2539fcf3e3846600c53
- c20 downstream analysis-stack hash: 0a00ab2ae00a468dcf4fb46ff825c8f5e0afefd27bd6f7457d4e214f7feb0d00

## Files

- c20-nuc-sensitivity.csv
- c20-nuc-sensitivity-draws.npz
- code/c20_nuc_sensitivity.py
