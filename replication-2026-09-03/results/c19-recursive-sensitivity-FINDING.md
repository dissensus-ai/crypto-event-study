# C19 -- Fixed-path versus floored recursive sensitivity

Both specifications use B=2,000 paired seed positions, the same union-calendar Student-t innovation vectors, and the same unrestricted refitter. The deliberate difference is whether the fitted null variance path is held fixed or propagated recursively with the stated positivity floor and explosion guard.

The retained machine label `crisis` denotes the asset-specific high-variance window from each asset's last 2021 conditional-variance break to its first break in 2022 or later; it is not an FTX-specific control.

| spec | retained fixed / recursive | one-sided p fixed / recursive | absolute-statistic p fixed / recursive | null mean fixed / recursive | null SD fixed / recursive | floor-hit draws |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2000 / 2000 | 0.3883 / 0.0255 | 0.3893 / 0.0600 | 1.2796 / -0.0151 | 0.8043 / 0.8915 | 40.2% |
| crisis | 1998 / 1999 | 0.3942 / 0.0405 | 0.3987 / 0.0900 | 1.1001 / -0.0054 | 0.9064 / 0.9089 | 46.1% |

The recursive p-values use add-one smoothing over retained draws. The reported worst-case bounds assign every excluded draw below or above the observed statistic; exact bounds, rescue counts, floor telemetry, and realised-variance diagnostics are in the CSV and seed-indexed NPZ files.

Floor-hit versus no-floor partitions are post hoc diagnostics. Differences between those strata do not identify a causal effect of flooring. The recursive result is therefore evidence under the implemented floored algorithm, not an unconstrained recursive GJR-GARCH-X bootstrap.

Files: `c19-recursive-sensitivity-{baseline,crisis}.csv` and `c19-recursive-draws-{baseline,crisis}.npz`; baseline aliases are byte-identical to the baseline-specific files.
