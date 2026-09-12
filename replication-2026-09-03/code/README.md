# Replication guide - accepted Digital Finance multi-moment event study

This directory is the self-contained analysis package for *Do Cryptocurrency
Markets Differentiate Infrastructure from Regulatory Shocks? A Multi-Moment
Event Study with Dependence-Robust Inference*.

The package layout is:

- `code/`: scripts c1-c21 and all local estimator modules;
- `code/data/`: the committed price, event, sentiment, and first-moment inputs;
- `results/`: committed numerical outputs and bootstrap draw archives;
- `figures-new/`: figures included by `main.tex`;
- `requirements.txt`: portable version ranges;
- `requirements-lock.txt`: the exact final-pass runtime pins;
- `main.tex`, `references.bib`, and the Springer Nature class/style files.

No script needs data from another checkout. Set `CES_OUT_DIR` only if outputs
should be written somewhere other than `results/`; set `CES_FIGURE_DIR` only to
override `figures-new/` for the heatmap.

## Environment

The final reproduction pass was run with CPython 3.14.0 on Linux x86-64. Use
`requirements-lock.txt` to reproduce that tested package stack; use
`requirements.txt` when portable version ranges are preferable.

The full parallel bootstrap stack requires a POSIX runtime with the
`multiprocessing` `fork` start method; the supported and tested target is Linux.
Windows is not a supported full-reproduction platform because the worker design
inherits fitted state through `fork`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`numba` is optional. `tarch_x_fast.py` falls back to its NumPy implementation
when numba is unavailable; the authoritative run used that fallback. `ruptures`
is required by c3. On the tested CPython 3.14 system it was installed from the
locally built `ruptures==1.1.9` wheel whose SHA-256 is recorded in the lockfile;
another platform may need to build the same version from source.

## Decision-relevant final results

- First moment: infrastructure CAR `-0.6395%`, regulatory CAR `-9.3326%`,
  difference `+8.6932` percentage points; event-block `p=0.2020`, event-level
  Welch `p=0.2164`.
- Curated second-moment point estimate: `3.4879x`.
- Fixed-path Student-t-copula bootstrap against the fitted sharp
  per-asset-equality null: one-sided `p=0.3883` and
  absolute-statistic `p=0.3893` at baseline.
  The machine-readable C9 column `p_tcopula_two_sided` is retained as a legacy
  schema name for that unrecentred absolute-statistic tail; it is not a generic
  two-sided p-value.
- The intermediate C8h Gaussian-control column `robust_p_two_sided` has the
  same legacy naming issue. The release verifier treats it as the unrecentred
  absolute-statistic tail and requires the C8h vectors to match C9's audited
  Gaussian legs exactly.
- Recursive sensitivity: baseline one-sided `p=0.02549`, absolute-statistic
  `p=0.05997`; asset-specific high-variance-control one-sided `p=0.04050`,
  absolute-statistic `p=0.09000`. The retained machine key `crisis` means the
  window from each asset's last 2021 conditional-variance break to its first
  2022-or-later break; four windows end around FTX, while BNB and ADA end in
  July 2022, so it is not an FTX-specific control.
- Paired copula-df sensitivity at `nu_c=3.8277/5.9/8.0`: one-sided
  `p=0.3883/0.3968/0.4013`, with `2000/2000` retained in every row. The value
  `5.9` is an alternative sensitivity value, not a rank-based estimate.
  C20's NPZ retains C9's whole-file source hash at run time. The packaged C9
  file differs only in its corrected human-facing finding renderer;
  `verify_release.py` pins both versions and separately proves that the archived
  numerical refitter and simulator contracts match the packaged analysis code.
- C10/C10b calibrate their bootstrap rungs with empirical percentile critical
  values. This is the finite-reference-sample analogue of the C7/C9 add-one
  upper-tail rule, not a literal identity at boundary draws; the committed size
  rates describe the implemented critical-value procedure.
- The honest second-moment verdict is directional, conditioning-scheme-sensitive,
  and unresolved.

## Ordered pipeline

The expensive bootstrap draw archives are committed. To regenerate everything,
run the existing scripts in this dependency order:

1. `c1_build_candidate_pool.py`
2. `c2_relaxed_threshold_sensitivity.py`
3. `c2b_two_asset_point.py` and `c2c_corrected_figure.py`
4. `c3_bai_perron.py` and `c4_granger_causality.py`
5. `c5_pseudoreplication_test.py` and `c6_garchx_clustered.py`
6. `c7_ccc_garchx_bootstrap.py`
7. `c8a`-`c8h` control scripts (`c8f` also writes the `c8g` FDR table)
8. `c9_tcopula_bootstrap.py`
9. `c10_size_study.py` and `c10b_dgp_perturbation.py`
10. `c11_returns_block_bootstrap.py`
11. `c12_granger_rigour.py` through `c16_nonoverlap_pre_dummy.py`
12. `c18_model_comparison_refit.py`, then `c17_model_comparison.py` to render its
    compact manuscript table
13. `c19_recursive_sensitivity.py baseline` and
    `c19_recursive_sensitivity.py crisis`
14. `c20_nuc_sensitivity.py` and `c21_events_in_mean.py`

Scripts c7, c9, c10, c13, c19, and c20 are computationally intensive. Seeds and
reported Monte-Carlo draw counts are fixed in their source files.

## Verification

Run the fast release checks after any change:

```bash
python code/verify_release.py
python -m compileall -q code
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The verifier checks the corrected Celsius UTC date, principal first- and
second-moment values, usable-draw counts, mechanical-screen verdict, script
inventory, and stale manuscript phrases. It intentionally verifies committed
bootstrap artifacts instead of recomputing multi-hour simulations.

## Data provenance

- Daily cryptocurrency price series: CoinGecko-derived committed CSVs.
- Event census: hand-curated public-record events in `code/data/events.csv` and the
  reconstructed candidate pool generated by c1.
- Weekly news-tone inputs: committed GDELT-derived `code/data/gdelt.csv`.
- First-moment smoke-test cache: committed Binance OHLCV parquet files.

For the daily variance models, the sentiment control is constructed on the
six-asset common calendar (beginning 2 September 2019) and then reindexed to
each asset. Consequently, the five longer histories zero-code 24 June through
1 September 2019 in addition to the rolling-z-score warm-up; BNB is unaffected.
This common-calendar alignment is disclosed as a nuisance-control limitation in
the manuscript. An audit-only union-calendar observed refit leaves the headline
multiplier at `3.49x` to reported precision; the bootstrap was not recalibrated,
so that check is not a separate inference result.

The unified analysis dates the Celsius withdrawal halt `2022-06-13`, following
the package's UTC announcement convention. Both the curated events file and the
regenerated candidate census use that date. The predecessor-reproduction file
`events_reclassified.json` intentionally retains its historical `2022-06-12`
local-date coding so the legacy negative-valence smoke test remains exact; it is
not used by the unified results.
