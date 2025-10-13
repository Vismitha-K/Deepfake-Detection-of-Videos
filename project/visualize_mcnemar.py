import os, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# === CONFIG ===
eval_dir = "multi_results"
input_file = os.path.join(eval_dir, "mcnemar_results.json")

# === LOAD DATA ===
with open(input_file, "r") as f:
    results = json.load(f)

df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
df["Significant"] = df["pvalue"] < 0.05

# === PLOT ===
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#1f77b4' if sig else '#ff7f0e' for sig in df["Significant"]]
bars = ax.bar(df["Model"], df["statistic"], color=colors, width=0.6)

ax.set_ylabel("McNemar Test Statistic", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_title("Statistical Significance via McNemar Test",
             fontsize=14, fontweight="bold", pad=15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.xticks(rotation=20, ha="center", fontsize=11)
plt.yticks(fontsize=11)

# === DYNAMIC TEXT PLACEMENT (p-values INSIDE bars)
for i, row in df.iterrows():
    height = row["statistic"]
    ax.text(i, height * 0.95, f"p={row['pvalue']:.1e}",
            ha="center", va="top", fontsize=10, color="white", fontweight="bold")

# === LEGEND ===
legend_handles = [
    Patch(color="#1f77b4", label="Statistically Significant (p < 0.05)"),
    Patch(color="#ff7f0e", label="Not Significant (p ≥ 0.05)")
]
ax.legend(handles=legend_handles, loc="upper right",
          frameon=False, fontsize=10)

# === LAYOUT ===
plt.tight_layout()
out_path = os.path.join(eval_dir, "mcnemar_significance_plot_inside.png")
plt.savefig(out_path, dpi=400, bbox_inches="tight")
plt.show()

print(f"✅ Saved clean McNemar plot with p-values inside to {out_path}")