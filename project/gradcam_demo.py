import os, sys, argparse, subprocess, shutil
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F

print(">>> Grad-CAM demo script started <<<")

# ----------------------------------------------------
# Ensure local imports work when running this file
# ----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ==========================================================
# Model Factory
# ==========================================================
def get_model(name, num_classes=2, device='cpu'):
    name = name.lower()
    from torchvision import models
    import torch

    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)

    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)

    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)

        # Some checkpoints were trained with 960 features instead of 1280
        try:
            in_f = m.classifier[-1].in_features
        except AttributeError:
            in_f = 960  # fallback

        # Use 960 intentionally to match training head
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(960, num_classes)
        )

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)

    elif name == "moe":
        # Import your hybrid model
        from moe_model import MoEModel
        m = MoEModel(
            resnet_ckpt=None,
            mobilenet_ckpt=None,
            device=device
        )
        print("[Info] Using hybrid Mixture-of-Experts (MoE) model.")

    else:
        raise ValueError(f"Unsupported model: {name}")

    return m

# ==========================================================
# Frame utilities
# ==========================================================
def preprocess_frame(img_bgr, size=299):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)


def extract_frames(video_path, every_n=5, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx, saved = 0, 0
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


def save_index_html(folder, thumbnails, title="Grad-CAM results"):
    html = f"<html><body><h2>{title}</h2>"
    for t in thumbnails:
        html += f"<div style='display:inline-block;margin:6px;text-align:center;'>"
        html += f"<img src='{os.path.basename(t)}' width=240><br>{os.path.basename(t)}</div>"
    html += "</body></html>"
    with open(os.path.join(folder, "index.html"), "w") as f:
        f.write(html)


# ==========================================================
# Main Grad-CAM analysis
# ==========================================================
def main(args):
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # -----------------------------
    # Load model and checkpoint
    # -----------------------------
    print(f"Loading model: {args.model}")
    model = get_model(args.model, device=device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("Loaded checkpoint successfully.")

    from gradcam import GradCAM, overlay_cam_on_image
    gradcam = GradCAM(model, use_cuda=(device.type == "cuda"))

    frames = extract_frames(args.video, every_n=args.every_n, max_frames=args.max_frames)
    print(f"Extracted {len(frames)} frames from {args.video}")

    results_folder = os.path.join(args.out, os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(results_folder, exist_ok=True)

    per_frame = []
    for i, (frame_index, frame_bgr) in enumerate(frames):
        inp = preprocess_frame(frame_bgr, size=args.input_size).to(device)
        with torch.no_grad():
            out = model(inp)
            logits = out['logits'] if isinstance(out, dict) else out
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_fake = float(probs[1])
        # For Grad-CAM, target class = predicted or fixed as FAKE (1)
        target_class = int(np.argmax(probs))
        cam = gradcam.generate_cam(inp, target_class=target_class)
        cam_energy = float(cam.mean())
        overlay = overlay_cam_on_image(frame_bgr, cam, alpha=args.alpha_overlay)
        out_name = f"{i:03d}_frame{frame_index}_p{prob_fake:.3f}_ce{cam_energy:.4f}.jpg"
        out_path = os.path.join(results_folder, out_name)
        cv2.imwrite(out_path, overlay)
        per_frame.append(out_path)

    save_index_html(results_folder, per_frame[:args.topk], title=f"Grad-CAM - {args.model}")
    print("Saved results to:", results_folder)
    print("--- Finished ---")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--model", required=True, help="model name: resnet50, densenet121, mobilenet_v3_large, efficientnet_b0")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pth file")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--every-n", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--alpha-overlay", type=float, default=0.5)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    main(args)