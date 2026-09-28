# Replication package (revision 8, 3 September 2026)

This directory contains the code, data, stored numerical outputs and dependency files from the 3 September 2026 revision 8 production version of *Do Cryptocurrency Markets Differentiate Infrastructure from Regulatory Shocks? A Multi-Moment Event Study with Dependence-Robust Inference*, forthcoming in *Digital Finance*.

Work from this directory, not the repository root. Read [`code/README.md`](code/README.md) for analysis commands and `requirements-lock.txt` for the original numerical environment. Use a project-local virtual environment. Long bootstrap runs can take substantial time; no API access is needed, because all analysis inputs are committed.

## Validation state

On 12 September 2026, `verify_release.py` passed against the full production package, including 87/87 tracked table values, using NumPy 2.3.5, SciPy 1.16.3 and pandas 2.3.3. This was a stored-artifact/implementation consistency verification, not a full rerun of every bootstrap. The default workstation numerical libraries failed a strict critical-value tolerance at roughly 3.1e-11; the matching numerical libraries passed without changing results or tolerances. On 28 September 2026, a clean clone verified against `PAYLOAD_SHA256SUMS`, and `c21_events_in_mean.py` rerun from it regenerated its stored output byte-for-byte.

The manuscript source is not redistributed here. The manuscript-dependent `verify_tables.py` / `verify_release.py` checks therefore need a copy of the corresponding `main.tex` placed in this directory; a missing-manuscript error from them is not a numerical failure. All analysis scripts, their inputs and their stored outputs are included.

`PAYLOAD_SHA256SUMS` identifies the payload. Source and output bytes are unchanged from revision 8. PNG/PDF presentation outputs and runtime caches are omitted; the scripts regenerate figures.

## Interpretation

The contribution is the documented correction of earlier significance claims and an inference/selection diagnostic toolkit. Fixed-path and floored recursive analyses support different inferential conclusions; neither the null nor a stable causal market asymmetry is established.
