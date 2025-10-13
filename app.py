import streamlit as st
import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import time
from project.gradcam import GradCAM, overlay_cam_on_image

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Deepfake Detection with Grad-CAM", layout="wide")
st.title("🎭 Deepfake Detection with Explainability")
st.write(
    "Upload a video and select a model to analyze whether it is **REAL or FAKE**, "
    "with **Grad-CAM** heatmaps highlighting regions influencing the prediction."
)

# -------------------------------
# Model Loading Function
# -------------------------------
@st.cache_resource
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        ckpt = os.path.join("checkpoints", "resnet50", "resnet50_best.pth")

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        ckpt = os.path.join("checkpoints", "densenet121", "densenet121_best.pth")

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        # Fix: match your training classifier structure
        in_features = 960  # the feature size used in your training
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features, 2)
        )
        ckpt = os.path.join("checkpoints", "mobilenet_v3_large", "mobilenet_v3_large_best.pth")

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        ckpt = os.path.join("checkpoints", "efficientnet_b0", "efficientnet_b0_best.pth")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not os.path.exists(ckpt):
        st.error(f"❌ Checkpoint not found at {ckpt}. Please ensure correct folder structure.")
        st.stop()

    # Load weights safely (strict=False for MobileNet)
    state_dict = torch.load(ckpt, map_location=device)
    if model_name == "mobilenet_v3_large":
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        st.warning(f"⚠️ Loaded MobileNetV3 checkpoint with relaxed matching. "
                   f"Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        model.load_state_dict(state_dict)

    model.to(device).eval()
    gradcam = GradCAM(model, use_cuda=(device.type == "cuda"))
    return model, gradcam, device

# -------------------------------
# Utility Functions
# -------------------------------
def preprocess_frame(frame_bgr, size=224):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def extract_frames(video_path, every_n=5, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append((idx, frame.copy()))
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

# -------------------------------
# Streamlit UI
# -------------------------------
model_choice = st.selectbox(
    "Select the model for analysis:",
    ("ResNet50", "DenseNet121", "EfficientNet_B0", "MobileNet_V3_Large")
)
st.markdown(f"**Selected Model:** {model_choice}")

uploaded_file = st.file_uploader("📂 Upload a video file (mp4/avi)", type=["mp4", "avi"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    input_path = os.path.join("uploads", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(input_path)
    st.success("✅ Video uploaded successfully!")

    # -------------------------------
    # Model Loading (cached)
    # -------------------------------
    with st.spinner(f"Loading {model_choice} model..."):
        model, gradcam, device = load_model(model_choice)

    # -------------------------------
    # Deepfake Analysis
    # -------------------------------
    with st.spinner("Analyzing video frames... Please wait ⏳"):
        frames = extract_frames(input_path, every_n=6, max_frames=80)
        st.write(f"Extracted {len(frames)} frames for analysis.")

        results_dir = os.path.join("results", os.path.splitext(uploaded_file.name)[0] + f"_{model_choice.lower()}")
        os.makedirs(results_dir, exist_ok=True)
        per_frame_scores = []

        for i, (frame_idx, frame) in enumerate(frames):
            inp = preprocess_frame(frame).to(device)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                prob_fake = float(probs[1])  # index 1 = "fake"

            cam = gradcam.generate_cam(inp, target_class=1)
            overlay = overlay_cam_on_image(frame, cam, alpha=0.5)
            out_name = f"{i:03d}_frame{frame_idx}_p{prob_fake:.3f}.jpg"
            out_path = os.path.join(results_dir, out_name)
            cv2.imwrite(out_path, overlay)
            per_frame_scores.append(prob_fake)

        avg_prob_fake = float(np.mean(per_frame_scores)) if per_frame_scores else 0.0
        verdict = "🟢 REAL" if avg_prob_fake > 0.5 else "🔴 FAKE"
        time.sleep(1)

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("📊 Detection Summary")
    st.markdown(f"### Final Verdict: **{verdict}**")
    st.markdown(f"**Average Real Probability:** {avg_prob_fake:.3f}")

    result_images = sorted([os.path.join(results_dir, f)
                            for f in os.listdir(results_dir) if f.endswith(".jpg")])[:6]
    if result_images:
        st.subheader("🧠 Top Frames with Grad-CAM Visualization")
        st.image(result_images, width=300, caption=[os.path.basename(i) for i in result_images])

    st.info("✅ Analysis complete. Grad-CAM visualizations saved in the 'results/' folder.")