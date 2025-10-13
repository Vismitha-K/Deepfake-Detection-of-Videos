import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to your selected Grad-CAM frames
images = {
    "ResNet50": "results/final_gradcam/resnet50.jpg",
    "EfficientNet-B0": "results/final_gradcam/efficientnet_b0.jpg",
    "DenseNet121": "results/final_gradcam/densenet121.jpg",
    "MobileNetV3-Large": "results/final_gradcam/mobilenet_v3_large.jpg"
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Grad-CAM Visualizations for Deepfake Detection", fontsize=16, fontweight="bold")

for ax, (title, path) in zip(axes.ravel(), images.items()):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/final_gradcam_comparison.png", dpi=300)
plt.show()