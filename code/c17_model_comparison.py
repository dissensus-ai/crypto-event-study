"""
C17: Regenerate the model-comparison table (manuscript Table 2) from the frozen
     model-selection JSONs — closing the one paper<->code gap with no active script.
================================================================================

The manuscript's model-comparison table (GARCH(1,1) vs GJR-GARCH vs GJR-GARCH-X
AIC/BIC/LogLik per asset) originates from the model-selection run archived in
`_archive/outputs/analysis_results/model_parameters/<asset>_parameters.json`.
Those JSONs are the source of record for the table; this script extracts and
tabulates them so the table is reproducible from committed artifacts like every
other table in the paper.

Parameter-count convention (matches the manuscript's ten-parameter statement):
the estimators profile the mean at the sample mean, so the GJR-GARCH-X
information criteria correspond to k = 10 (omega, alpha, gamma, beta, nu + 5
exogenous coefficients); BIC - AIC = k * (ln n - 2). LogLik is recovered from
AIC as (2k - AIC) / 2.

Output (results/):
    c17-model-comparison.csv    one row per (asset, model): AIC, BIC, LogLik
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JSON_DIR = ROOT / "_archive" / "outputs" / "analysis_results" / "model_parameters"
OUT_DIR = ROOT / "results"

ASSETS = ["btc", "eth", "xrp", "bnb", "ltc", "ada"]
# model label in JSON -> (manuscript label, parameter count with mean profiled)
MODELS = {
    "GARCH(1,1)": ("GARCH(1,1)", 5),
    "TARCH(1,1)": ("GJR-GARCH", 6),
    "TARCH-X": ("GJR-GARCH-X", 10),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for asset in ASSETS:
        with open(JSON_DIR / f"{asset}_parameters.json") as f:
            data = json.load(f)
        for json_label, (paper_label, k) in MODELS.items():
            block = data[json_label]
            aic, bic = float(block["AIC"]), float(block["BIC"])
            loglik = (2 * k - aic) / 2
            rows.append(
                {
                    "asset": asset.upper(),
                    "model": paper_label,
                    "k_params": k,
                    "AIC": round(aic, 3),
                    "BIC": round(bic, 3),
                    "loglik": round(loglik, 3),
                }
            )
    out = OUT_DIR / "c17-model-comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    # console preview mirroring the manuscript table layout
    for asset in ASSETS:
        sub = [r for r in rows if r["asset"] == asset.upper()]
        best = min(sub, key=lambda r: r["AIC"])
        line = "  ".join(
            f"{r['model']}: AIC {r['AIC']:.0f} BIC {r['BIC']:.0f} LL {r['loglik']:.0f}"
            + (" *" if r is best else "")
            for r in sub
        )
        print(f"{asset.upper():4s} {line}")


if __name__ == "__main__":
    main()
