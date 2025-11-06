# visualize_results_balanced_bar.py
import matplotlib.pyplot as plt
import pandas as pd
import os, json

# === CONFIG ===
eval_dir = "../multi_results"

# === LOAD METRICS ===
files = [f for f in os.listdir(eval_dir) if f.endswith("_metrics.json")]
data = []
for f in files:
    model = f.replace("_metrics.json", "")
    with open(os.path.join(eval_dir, f), "r") as fh:
        m = json.load(fh)
        data.append([model, m["accuracy"], m["precision"], m["recall"], m["f1"]])

df = pd.DataFrame(data, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
df.set_index("Model", inplace=True)

# === ORDER MODELS (Hybrid first) ===
order = [
    "proposed_moe",  # now called hybrid_model
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large"
]
df = df.reindex(order).dropna()

# === RENAME FOR LABEL ===
df.rename(index={"proposed_moe": "hybrid_model"}, inplace=True)

# === PLOT COLORFUL BAR GRAPH ===
fig, ax = plt.subplots(figsize=(10, 6))

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

df.plot(kind="bar", color=colors, width=0.75, ax=ax, edgecolor='black', linewidth=0.7)

# === TITLES & LABELS ===
ax.set_title(
    "Comparative Performance of CNN Architectures and Hybrid (MoE) Model",
    fontsize=14,
    fontweight='bold',
    pad = 30 #balanced title spacing
)
ax.set_ylabel("Metric Value", fontsize=12)
ax.set_xlabel("Model Architecture", fontsize=12)
ax.set_ylim(0.85, 1.00)  # start at 0.85 to balance proportions
ax.grid(axis="y", linestyle="--", alpha=0.4)

# === LEGEND (moved slightly down & resized) ===
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.10),  # slightly lower for separation
    ncol=4,
    frameon=False,
    fontsize=10.5,
    columnspacing=1.4,
    handletextpad=0.5
)

# === AESTHETICS ===
plt.xticks(rotation=15, ha="center", fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])  # provides more top margin

# === SAVE ===
out_path = os.path.join(eval_dir, "bar_metrics_comparison_hybrid_balanced.png")
plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor='white')
plt.show()

print(f"✅ Balanced, clean figure saved to {out_path}")