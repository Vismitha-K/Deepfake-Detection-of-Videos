import matplotlib.pyplot as plt
import pandas as pd
import os, json

# === CONFIG ===
eval_dir = "multi_results"

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

# === PLOT ===
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
df.plot(kind="bar", color=colors, width=0.75, ax=ax)

# === TITLE & LABELS ===
ax.set_title(
    "Comparative Performance of CNN Architectures on Deepfake Detection",
    fontsize=14,
    fontweight='bold',
    pad=35  # reduced padding since legend is below title
)
ax.set_ylabel("Metric Value", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylim(0.85, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# === LEGEND POSITION: Just Below the Title ===
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.12),  # slightly below previous version
    ncol=4,
    frameon=False,
    fontsize=11,
    columnspacing=1.5,
    handletextpad=0.5
)

# === Aesthetic Adjustments ===
plt.xticks(rotation=20, ha="center", fontsize=11)
plt.yticks(fontsize=11)
plt.subplots_adjust(top=0.83, bottom=0.12)
plt.tight_layout(rect=[0, 0, 1, 0.94])

# === SAVE ===
out_path = os.path.join(eval_dir, "all_metrics_comparison_clean_spacing.png")
plt.savefig(out_path, dpi=400, bbox_inches="tight")
plt.show()

print(f"✅ Clean final figure saved to {out_path}")