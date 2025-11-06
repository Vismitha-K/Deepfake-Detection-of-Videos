import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
eval_dir = "../multi_results"
csv_files = [f for f in os.listdir(eval_dir) if f.endswith("_per_frame.csv")]

# Load all predictions
data = {}
for f in csv_files:
    model = f.replace("_per_frame.csv", "")
    df = pd.read_csv(os.path.join(eval_dir, f))
    data[model] = (df["label"] == df["pred"]).astype(int).values

# Align sample lengths
min_len = min(len(v) for v in data.values())
for k in data:
    data[k] = data[k][:min_len]

# Compute accuracies
accuracies = {m: np.mean(v) for m, v in data.items()}

# Sort: keep baseline models first, proposed model last
df_plot = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
df_plot = df_plot.sort_values(by="Accuracy", ascending=True)

# Move proposed_moe to the end
if "proposed_moe" in df_plot["Model"].values:
    hybrid_row = df_plot[df_plot["Model"] == "proposed_moe"]
    df_plot = pd.concat([df_plot[df_plot["Model"] != "proposed_moe"], hybrid_row])

# === IEEE STYLE ===
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))

# IEEE color palette (grayscale-friendly)
colors = ["#4C4C4C", "#7F7F7F", "#999999", "#BFBFBF", "#C00000"]  # last = highlight for hybrid

bars = plt.bar(df_plot["Model"], df_plot["Accuracy"], color=colors[:len(df_plot)])

# Highlight the hybrid model in dark red
for bar, model in zip(bars, df_plot["Model"]):
    if "proposed" in model:
        bar.set_color("#C00000")  # dark IEEE red

# Titles and labels (IEEE minimalism)
plt.title("Model-wise Accuracy Comparison (Cochran’s Q Test Results)", fontsize=13, fontweight="bold")
plt.ylabel("Accuracy", fontsize=11)
plt.xlabel("Model", fontsize=11)
plt.ylim(0.85, 1.0)
plt.xticks(rotation=20, fontsize=10)
plt.yticks(fontsize=10)

# Clean layout
sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])

# === SAVE ===
out_path = os.path.join(eval_dir, "cochran_q_results_ieee_style.png")
plt.savefig(out_path, dpi=400, bbox_inches="tight")
plt.show()

print(f"✅ Saved IEEE-standard figure to {out_path}")