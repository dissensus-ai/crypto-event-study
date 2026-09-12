"""
Corrected C2 figure + summary CSV (scope-condition framing).
Separates the primary curated sample from the nested mechanical impact-screen
sweep on the reconstructed 135-pool, and marks statistical significance.
Numbers: c2-summary-table.csv (curated/1-asset/no-filter/3-asset) plus
c2b-two-asset-result.csv (2-asset). No values are hard-coded.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CES_OUT_DIR", HERE.parent / "results"))
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(OUT / "c2-summary-table.csv").set_index("spec")
two = pd.read_csv(OUT / "c2b-two-asset-result.csv").iloc[0]

def row(label, spec, kind):
    r = summary.loc[spec]
    return (label, int(r.n_infra_events + r.n_reg_events), r.mean_infra_coef,
            r.mean_reg_coef, r.multiplier, r.welch_p, kind)

rows = [
    row("Curated\n(n=50)", "S1_baseline", "curated"),
    row("No filter\n(135)", "S3_nofilter", "mech"),
    row(r"$\geq$1 asset" + "\n(115)", "S2_relaxed", "mech"),
    (r"$\geq$2 assets" + f"\n({int(two.n_infra_events + two.n_reg_events)})",
     int(two.n_infra_events + two.n_reg_events), two.mean_infra_coef,
     two.mean_reg_coef, two.multiplier, two.welch_p, "mech"),
    row(r"$\geq$3 assets" + "\n(78)", "S4_strict", "mech"),
]
df = pd.DataFrame(rows, columns=["label", "n", "mean_infra", "mean_reg",
                                 "multiplier", "welch_p", "kind"])
df_csv = df.drop(columns="label").copy()
df_csv.insert(0, "spec", ["primary_curated", "nofilter_135", "oneasset_115",
                           "twoasset_94", "threeasset_78"])
df_csv["note"] = ["primary curated estimate; naive Welch p shown only for comparison",
                  "not significant", "not significant",
                  "like-for-like analogue of curated criterion; not significant",
                  "not significant"]
df_csv.to_csv(OUT / "c2-summary-CORRECTED.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 4.6))
colors = ["#7a1f3d" if k=="curated" else "#c8b58c" for k in df["kind"]]
bars = ax.bar(df["label"], df["multiplier"], color=colors, edgecolor="black", linewidth=0.6)
for b, p, m in zip(bars, df["welch_p"], df["multiplier"]):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.08, f"{m:.2f}x\n({sig})",
            ha="center", va="bottom", fontsize=9)
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_ylabel(r"$\bar{\delta}_{\mathrm{infra}}/\bar{\delta}_{\mathrm{reg}}$ multiplier")
ax.set_title("Infrastructure/regulatory multiplier: curated sample vs nested mechanical screens\n"
             "(Welch p-values shown for descriptive comparability)", fontsize=10)
ax.set_ylim(0, 5.6)
# divider between curated (distinct selection process) and the mechanical sweep
ax.axvline(0.5, color="black", lw=0.8, ls=":")
ax.text(3.0, 5.25, "mechanical impact screen on reconstructed 135-pool (nested, all n.s.)",
        ha="center", fontsize=8, color="#5a5a3d")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "c2-multiplier-CORRECTED.png", dpi=150)
fig.savefig(OUT / "c2-multiplier-CORRECTED.pdf")
print("wrote c2-summary-CORRECTED.csv, c2-multiplier-CORRECTED.{png,pdf}")
print(df_csv.to_string(index=False))
