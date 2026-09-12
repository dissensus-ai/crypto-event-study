"""
Generate the correlation heatmap requested by Reviewer 2 (R2.5).
6x6 daily-return correlation matrix with hierarchical clustering.
Saves PDF and PNG to the package figures-new directory.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # legacy event-study/ working-tree fallback
DATA_DIR = HERE / "data" if (HERE / "data").is_dir() else ROOT / "code" / "data"
# Output location. The package default matches main.tex's graphicspath; override
# with CES_FIGURE_DIR when a different build directory is required.
import os as _os
OUT_DIR = Path(_os.environ.get("CES_FIGURE_DIR", HERE.parent / "figures-new"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["BTC", "ETH", "XRP", "BNB", "LTC", "ADA"]


def load_returns():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(DATA_DIR / f"{a.lower()}.csv")
        df["date"] = pd.to_datetime(df["snapped_at"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        df = df.loc["2019-01-01":"2025-08-31"]
        out[a] = np.log(df["price"]).diff().dropna()
    return pd.DataFrame(out).dropna()


def main():
    rets = load_returns()
    corr = rets.corr()

    # Hierarchical clustering on the correlation distance
    dist = 1 - corr.abs()
    link = linkage(dist.values[np.triu_indices_from(dist, k=1)], method="average")
    order = leaves_list(link)
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Pearson correlation"},
                linewidths=0.5, ax=ax)
    ax.set_title("Daily log-return correlation matrix\n(Jan 2019 -- Aug 2025)",
                 pad=12, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_correlation_heatmap.pdf", dpi=200)
    fig.savefig(OUT_DIR / "figure_correlation_heatmap.png", dpi=200)
    print(f"Saved {OUT_DIR/'figure_correlation_heatmap.pdf'}")
    print(f"Saved {OUT_DIR/'figure_correlation_heatmap.png'}")


if __name__ == "__main__":
    main()
