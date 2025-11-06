# app.py — Streamlit Web App for Hybrid MoE Deepfake Detection

import streamlit as st
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import time
from torchvision import transforms
from project.gradcam import GradCAM, overlay_cam_on_image
from project.moe_model import MoEModel  # ✅ import your hybrid model

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Hybrid Deepfake Detection", layout="wide")
st.title("🎭 Hybrid Deepfake Detection with Explainability")
st.write(
    "Upload a video and let the **Hybrid Mixture-of-Experts (MoE)** model detect whether it is **REAL** or **FAKE**. "
    "The app also generates **Grad-CAM heatmaps** showing which regions influenced the decision."
)

# -------------------------------
# Cached Model Loader
# -------------------------------
@st.cache_resource
def load_hybrid_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Hybrid MoE model
    model = MoEModel(
        resnet_ckpt=None,
        mobilenet_ckpt=None,
        device=device
    )

    ckpt_path = os.path.join("checkpoints", "moe_finetuned.pth")
    if not os.path.exists(ckpt_path):
        st.error(f"❌ Checkpoint not found at {ckpt_path}. Please upload it to the 'checkpoints/' directory.")
        st.stop()

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
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


def extract_frames(video_path, every_n=5, max_frames=150):
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
# File Upload Section
# -------------------------------
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
    # Load Hybrid Model
    # -------------------------------
    with st.spinner("Loading Hybrid MoE Model..."):
        model, gradcam, device = load_hybrid_model()

    # -------------------------------
    # Frame Analysis
    # -------------------------------
    with st.spinner("Analyzing video frames... Please wait ⏳"):
        frames = extract_frames(input_path, every_n=6, max_frames=80)
        st.write(f"Extracted **{len(frames)}** frames for analysis.")

        results_dir = os.path.join("results", os.path.splitext(uploaded_file.name)[0] + "_hybrid")
        os.makedirs(results_dir, exist_ok=True)
        per_frame_scores = []

        for i, (frame_idx, frame) in enumerate(frames):
            inp = preprocess_frame(frame).to(device)
            with torch.no_grad():
                output = model(inp)
                logits = output["logits"] if isinstance(output, dict) else output
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
        st.subheader("🧠 Grad-CAM Visual Explanations")
        st.image(result_images, width=300, caption=[os.path.basename(i) for i in result_images])

    st.info("✅ Analysis complete. Grad-CAM visualizations saved in the 'results/' folder.")