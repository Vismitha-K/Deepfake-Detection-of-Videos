import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# === Paths to selected Grad-CAM frames ===
images = {
    "Hybrid Model": "../gradcam/hybrid.jpg",
    "ResNet50": "../gradcam/resnet.jpg",
    "DenseNet121": "../gradcam/densenet.jpg",
    "MobileNetV3-Large": "../gradcam/mobilenet.jpg",
    "EfficientNet-B0": "../gradcam/efficientnet.jpg"
}

# === Create clean, balanced layout ===
fig = plt.figure(figsize=(9, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.25, wspace=0.06)

fig.suptitle("Grad-CAM Visualizations of CNN Architectures for Deepfake Detection",
             fontsize=15, fontweight="bold", y=0.97)

# --- Row 1 ---
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
# --- Row 2 ---
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])
# --- Row 3 (Hybrid spans both columns) ---
ax5 = plt.subplot(gs[2, :])

# === Plot baseline models ===
rows = [
    ("ResNet50", ax1),
    ("MobileNetV3-Large", ax2),
    ("DenseNet121", ax3),
    ("EfficientNet-B0", ax4)
]

for title, ax in rows:
    img = mpimg.imread(images[title])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)

# === Plot hybrid model centered ===
img_hybrid = mpimg.imread(images["Hybrid Model"])
ax5.imshow(img_hybrid)
ax5.axis("off")
ax5.set_title("Hybrid Model (Proposed)", fontsize=12, fontweight="bold", pad=6)

# === Adjust spacing & save ===
plt.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.04, hspace=0.3)
output_path = "../results/final_gradcam_comparison_centered.png"
plt.savefig(output_path, dpi=400, bbox_inches="tight")
plt.show()

print(f"✅ Saved balanced Grad-CAM figure at: {output_path}")