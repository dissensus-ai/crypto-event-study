# Do Cryptocurrency Markets Differentiate Infrastructure from Regulatory Shocks?
### A Multi-Moment Event Study with Dependence-Robust Inference

Replication materials for the merged multi-moment paper (supersedes the earlier
single-moment "Infrastructure vs Regulatory Shocks" and the companion "Same Returns,
Different Risks").

**Finding.** Cryptocurrency markets do **not** statistically differentiate infrastructure
from regulatory shocks at *either* moment — returns (block-bootstrap *p* ≈ 0.28) or
conditional variance (Student-*t*-copula bootstrap *p* = 0.322) — once inference accounts
for cross-event/cross-asset dependence and heavy tails. The apparent volatility asymmetry
reported in the earlier version was an **inference artefact**. The lead contribution is a
methodological cautionary tale: an *inference ladder* and a *Monte-Carlo size study* showing
how naive event-study inference manufactures significance that correct inference dissolves.

## Repository structure
- `paper/` — merged manuscript (`main.tex`, `main.pdf`) + response/cover letters
- `code/` — verified analysis pipeline (`c1`–`c14`, `tarch_x_manual.py`, `tarch_x_fast.py`)
- `code/src/` — CAR engine for the returns leg (`ConstantMeanModel` in `event_study.py` + `config.py`), inherited from the retired standalone returns paper; imported by `c11`
- `data/` — shared sample: 6 assets (BTC, ETH, XRP, BNB, LTC, ADA), 50 events (`events.csv` + `events_reclassified.json`), GDELT sentiment
- `results/` — committed outputs (CSVs + per-analysis FINDING docs)
- `_archive/` — superseded prior-version materials; **do not cite**
- `springer-submission/` — frozen original single-moment submission, retained as record (superseded)

## Script → paper map
| script | produces |
|---|---|
| `c1_build_candidate_pool` | candidate-event pool + drop-out census |
| `c2_relaxed_threshold` / `c2b_two_asset_point` | scope-condition multiverse (curated vs mechanical) |
| `c3_bai_perron` | structural breaks (descriptive) |
| `c5_pseudoreplication_test` | inference-ladder rungs 2–3 (event-level / cluster) |
| `c6_garchx_clustered` | design-effect / correlation-weighted (rung 4) |
| `c7_ccc_garchx_bootstrap` | Gaussian-copula bootstrap (rung 5) |
| `c9_tcopula_bootstrap` | **Student-*t*-copula bootstrap — inference of record (rung 6)** |
| `c8a` / `c8h_break_controls` | structural-break regime controls |
| `c8b`/`c8c`/`c8d`/`c8e` | anticipation / winsorisation / constant-mean / persistence |
| `c8f_weekly_granger_fdr` | weekly sentiment-leads-volatility (+ BH-FDR) |
| `c10_size_study` | **Monte-Carlo size-distortion study** |
| `c11_returns_block_bootstrap` | first-moment returns null (rung 7) |

## Reproduce
Tested on **Python 3.11–3.13**. (Python 3.14 is not yet supported by the pinned `pandas==2.3.1`, whose datetime C-extension segfaults there — use 3.11–3.13.)
```
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python code/descriptive_stats.py       # Table 1 (fast sanity check)
python code/c9_tcopula_bootstrap.py    # inference of record (variance)
python code/c11_returns_block_bootstrap.py   # first moment (returns)
python code/c10_size_study.py          # Monte-Carlo size study
```
Scripts read the committed `data/*.csv` and write to `results/`; all seeds are
fixed, so a clean-clone run regenerates the committed CSVs (e.g. `c11` rewrites
`results/c-gate-returns-unified-results.csv` — `git diff` should come back clean).

**Returns leg (`c11`) in full:** the gate runs A/B — the numbers used in the paper
(block-bootstrap *p* = 0.283 on the 6-asset basis) — need only the committed CSVs
and run from a clean clone as-is. The script's optional *smoke test* additionally
replays the retired returns paper's original 4-asset sample, which requires a
Binance daily-kline cache that is **not** committed (derived data, `*.parquet` is
gitignored). Rebuild it first with
```
python code/fetch_binance_cache.py
```
which fetches from the public Binance klines API (no key needed), writes
`data/cache/*.parquet`, and verifies the rebuilt returns series against SHA-256
fingerprints of the original run. Without the cache, `c11` skips the smoke test
with a notice and still produces the gate results.

## Note on the prior version
This repository previously hosted the single-moment "5.7×, *p* = 0.0008" result. That
estimate did not survive dependence-robust inference; the point estimate is unchanged but it
is no longer statistically distinguishable from zero. This paper reports the corrected dual
null and the inference lesson openly, as self-correction. Superseded materials are in
`_archive/` and `springer-submission/`.

## License
- **Code** (`code/`, including `code/src/`): MIT — see `LICENSE`.
- **Manuscript** (`paper/`) and the FINDING docs in `results/`: CC BY 4.0.
- **Data**: `events.csv` / `events_reclassified.json` are author-curated (CC BY 4.0);
  price CSVs derive from CoinGecko public data and `gdelt.csv` from the GDELT
  Project — both redistributed here as small research extracts with attribution.
  The Binance-derived smoke-test cache is not redistributed; rebuild it locally
  with `code/fetch_binance_cache.py`.

## Citation
See `CITATION.cff`.
