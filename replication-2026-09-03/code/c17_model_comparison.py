"""C17: render the manuscript model-comparison table from the controlled c18 refit.

The earlier version read archived JSONs outside the replication package. The
canonical source is now the committed/reproducible c18 controlled-refit CSV.
This script extracts the conventional count that includes the profiled sample
mean (k=5/6/11) into a compact table. Counting the mean adds the same penalty to
all three models and does not change any within-asset selection verdict.
"""
import os
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = Path(os.environ.get("CES_OUT_DIR", ROOT / "results"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = OUT_DIR / "c18-model-comparison-refit.csv"
    df = pd.read_csv(source)
    out_df = pd.DataFrame({
        "asset": df["asset"].str.upper(),
        "model": df["model"],
        "k_params": df["k_mu"].astype(int),
        "AIC": df["AIC_kmu"].round(3),
        "BIC": df["BIC_kmu"].round(3),
        "loglik": df["LL"].round(3),
    })
    out = OUT_DIR / "c17-model-comparison.csv"
    out_df.to_csv(out, index=False)
    print(f"wrote {out} ({len(out_df)} rows) from {source.name}")
    # console preview mirroring the manuscript table layout
    for asset, sub in out_df.groupby("asset", sort=False):
        best_idx = sub["AIC"].idxmin()
        line = "  ".join(
            f"{r.model}: AIC {r.AIC:.0f} BIC {r.BIC:.0f} LL {r.loglik:.0f}"
            + (" *" if idx == best_idx else "")
            for idx, r in sub.iterrows()
        )
        print(f"{asset:4s} {line}")


if __name__ == "__main__":
    main()
