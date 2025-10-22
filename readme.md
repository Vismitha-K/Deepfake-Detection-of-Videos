# 🎭 Deepfake Detection with Explainability

This project implements a **deepfake video detection system** using multiple **CNN architectures** — ResNet-50, DenseNet-121, EfficientNet-B0, and MobileNetV3-Large — combined with **Grad-CAM** to provide visual explainability for each prediction.  
It detects whether a given video is **REAL or FAKE** and highlights manipulated facial regions.

---

## 🧠 Overview

- **Dataset:** Celeb-DF (v2)  
- **Goal:** Frame-based deepfake detection with model interpretability  
- **Explainability:** Grad-CAM visual heatmaps for spatial reasoning  
- **Frontend:** Streamlit web app for upload, prediction, and visualization  

---

## ⚙️ Features

- Multi-model training and evaluation (`train_multi.py`, `evaluate_multi.py`)  
- Grad-CAM explainability (`gradcam.py`, `gradcam_demo.py`)  
- Streamlit interface (`app.py`) for interactive demo  
- Statistical comparison using McNemar’s test  
- Visualization utilities for results and metrics  

---

## 📊 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|----------|
| ResNet-50 | **98.36%** | 96.98% | 90.84% | **93.82%** |
| EfficientNet-B0 | 97.92% | 96.11% | 88.30% | 92.04% |
| DenseNet-121 | 97.79% | 95.51% | 87.96% | 91.58% |
| MobileNetV3-Large | 97.92% | 96.11% | 88.30% | 92.04% |

**Best Model:** ResNet-50 (used in Streamlit demo)

---

## 🚀 Usage

```bash
# Clone repository
git clone https://github.com/Vismitha-K/deepfake-detection-videos.git
cd deepfake-detection-videos

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run Streamlit demo
streamlit run app.py
````
Upload a `.mp4` video and view **REAL/FAKE prediction** with **Grad-CAM overlays**.

---

## 💡 Acknowledgements

* Celeb-DF (v2) dataset
* Arman176001/deepfake-detection (baseline reference)
* TorchVision & Streamlit libraries

