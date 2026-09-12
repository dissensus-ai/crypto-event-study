# September production replication

This directory contains the code, data, stored numerical outputs and dependency files from the 3 September 2026 revision 8 production resupply of *Do Cryptocurrency Markets Differentiate Infrastructure from Regulatory Shocks? A Multi-Moment Event Study with Dependence-Robust Inference*.

Work from this directory, not the repository root. Read [`code/README.md`](code/README.md) for analysis commands and `requirements-lock.txt` for the original numerical environment. Use a project-local virtual environment. Long bootstrap runs can take substantial time; no API access is needed for the stored analysis inputs.

## Validation state

On 12 September, `verify_release.py` passed against the full local production package, including 87/87 tracked table values, using NumPy 2.3.5, SciPy 1.16.3 and pandas 2.3.3. This was a stored-artifact/implementation consistency verification, not a full rerun of every bootstrap. The default workstation numerical libraries failed a strict critical-value tolerance at roughly 3.1e-11; the matching numerical libraries passed without changing results or tolerances.

The manuscript source is deliberately not copied into this public code branch while the article-specific redistribution agreement is being checked. Consequently the manuscript-dependent `verify_tables.py` / `verify_release.py` checks require placing the separately supplied corresponding `main.tex` in this directory. Analysis scripts and their stored inputs/outputs are included. Do not interpret a missing manuscript-file error as a numerical failure.

`PAYLOAD_SHA256SUMS` identifies the copied payload. Source and output bytes are unchanged from revision 8. PNG/PDF presentation outputs and runtime caches are omitted; scripts can regenerate figures. The parent branch preserves historical files for comparison, and no main-branch merge is implied.

## Interpretation

The contribution is the documented correction of earlier significance claims and an inference/selection diagnostic toolkit. Fixed-path and floored recursive analyses support different inferential conclusions; neither the null nor a stable causal market asymmetry is established.
