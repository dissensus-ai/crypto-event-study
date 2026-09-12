# C8h — Intermediate Gaussian-Innovation Break-Control Bootstrap

_Null-imposed CCC-GARCH-X parametric bootstrap, B=2000; runtime 42.8 min; same engine as c7 (regime dummies added to the variance-equation exog)._

This is a retained Gaussian-innovation comparator. The final matched-margin inference is c9; a c8h run with more than 10% discarded draws is reliability-flagged.

## Result

| variant | multiplier | naive Welch p | **robust CCC p (1-sided)** | robust absolute-statistic p | rho_resid | dropped |
|---|---|---|---|---|---|---|
| full regime | 1.94x | 0.0303 | **0.2404** | 0.2519 | 0.700 | 0.0% |
| asset-specific high-variance regime | 2.90x | 0.0124 | **0.1381** | 0.1386 | 0.702 | 0.1% |

- **full regime dummies** (multiplier 1.94x, d_bar_obs=1.216): Not below 10% in this intermediate Gaussian-innovation bootstrap (p=0.2404). Naive Welch was 0.0303; the cross-asset-robust value is 0.2404.
- **asset-specific high-variance regime dummies** (multiplier 2.90x, d_bar_obs=1.296): Not below 10% in this intermediate Gaussian-innovation bootstrap (p=0.1381). Naive Welch was 0.0124; the cross-asset-robust value is 0.1381.

## Honest verdict

Gaussian-innovation one-sided p-values are full=0.2404 and asset-specific high-variance=0.1381 (legacy machine key `crisis`). They are intermediate diagnostics and must not replace c9's Student-t results.

## Files

- `c8h-break-ccc-bootstrap-results.csv`, `c8h-break-ccc-draws-{full,crisis}.npz`
- `code/c8h_break_controls_ccc_bootstrap.py` (reuses `c7_ccc_garchx_bootstrap.py` engine)

## Release provenance

- C8h source SHA-256: `ea50efe3a34c14cdcb0b76dcb80ba9dea3c9fa32bec2c92da7cd5531cb3aaf6f`
- C8h results CSV SHA-256: `d219a21c38c783a25510b2b66aa210da2525195222021c22b4fd74171cd7f248`
- Full C8h/C9 draw-archive SHA-256: `22443ade24dd51b19d1e9284f80f7e7caeb3fb9762fa9741ed214080a9715f98` / `5735c9bd8dffaeaa15ad5c74fbd4746d527544dbe976710db44a716dfe73549b`
- Asset-specific high-variance C8h/C9 draw-archive SHA-256: `fef6d1ec302d4c21fa15112bdb00cfb5e6c3c0a64d5d85b99a4e6e87be8b0c92` / `2376f78397811c85d48b6d7af8138c4daad9f1300e22ea12de638c231b971644`
The release verifier recomputes all C8h tail counts and summaries and requires its retained vectors, observed statistics, calendar, and availability patterns to match C9's corresponding Gaussian legs exactly.
The legacy CSV field `robust_p_two_sided` stores the unrecentred absolute-statistic tail; it is not a generic two-sided p-value.