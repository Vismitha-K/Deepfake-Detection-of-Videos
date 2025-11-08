# 🎭 Hybrid Explainable Deepfake Detection using Mixture-of-Experts (MoE) and Grad-CAM

### 🔍 Overview
This project presents a **Hybrid Mixture-of-Experts (MoE) framework** for **Deepfake Video Detection** that combines the representational strengths of multiple CNN architectures and integrates **explainable AI (XAI)** through **Grad-CAM** visualization.  
The system not only classifies videos as *Real* or *Fake* but also highlights the manipulated regions of the face to ensure interpretability and forensic reliability.

---

### 🧠 Motivation
Deepfakes—synthetic videos generated using deep generative models like GANs—pose significant threats to privacy, media authenticity, and digital trust.  
Most existing detectors rely on single CNN architectures that are accurate but opaque and fail to generalize across manipulation types.  
This project proposes a **hybrid deepfake detector** that is both **accurate and explainable**, bridging the gap between performance and transparency.

---

### 🎯 Objectives
- To design a **robust and explainable** deepfake detection system.  
- To combine multiple CNN architectures into a **Mixture-of-Experts (MoE)** model for improved accuracy and generalization.  
- To integrate **Grad-CAM** visualization for **explainable AI**, highlighting regions influencing model predictions.  
- To develop an **interactive Streamlit-based interface** for real-time video analysis.

---

### ⚙️ Architecture
#### 🧩 Hybrid Mixture-of-Experts (MoE)
- **Experts:** ResNet-50, DenseNet-121, EfficientNet-B0, and MobileNetV3-Large.  
- **Gating Network:** Learns dynamic weights (e.g., ResNet 60%, MobileNet 40%) for expert contributions.  
- **Classifier Head:** Aggregates expert features for final binary classification (Real vs Fake).  
- **Explainability:** Grad-CAM overlays highlight manipulated regions such as mouth, eyes, and jawline.

---

### 📊 Dataset
**Celeb-DF (v2)** dataset was used for training and evaluation.  
- Total videos: 6,500 (590 real, 5,939 fake)  
- ~52,000 frames extracted using OpenCV (`cv2.VideoCapture`)  
- Train/Validation Split: 80/20  
- Image size: `224x224`, normalized with ImageNet mean and standard deviation.  
- Augmentations: random horizontal flip, rotation.

---

### 🧪 Experimental Setup
| Configuration | Details |
|----------------|----------|
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 32 |
| Epochs | 8 |
| Hardware | Google Colab (Tesla T4 GPU, 16GB VRAM) |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-score |

---

### 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| **Hybrid MoE (Proposed)** | **0.9936** | **0.9947** | **0.9932** | **0.9940** |
| ResNet-50 | 0.9837 | 0.9698 | 0.9084 | 0.9381 |
| DenseNet-121 | 0.9779 | 0.9551 | 0.8796 | 0.9158 |
| EfficientNet-B0 | 0.9792 | 0.9611 | 0.8830 | 0.9204 |
| MobileNetV3-Large | 0.9787 | 0.9648 | 0.8757 | 0.9181 |

📌 The **Hybrid MoE** model achieved the **highest accuracy and interpretability**, outperforming all baselines.

---

### 🔥 Explainability — Grad-CAM Visualizations
Grad-CAM (Gradient-weighted Class Activation Mapping) was applied on hybrid model predictions to visualize focus regions.  
The heatmaps consistently emphasize the **mouth, jawline, and eyes**, confirming that the model detects semantically meaningful artifacts rather than random textures.

---

### 📊 Statistical Validation
To validate model differences, **Cochran’s Q Test** was used to assess statistical significance across CNN architectures.  
The hybrid model showed consistent superiority (p < 0.05), indicating meaningful performance improvement beyond random variation.

---

### 🚀 Usage
```bash
# Clone repository
git clone https://github.com/Vismitha-K/Deepfake-Detection-of-Videos.git
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

### 💡 Acknowledgements
* Celeb-DF (v2) dataset
* Arman176001/deepfake-detection (baseline reference)
* TorchVision & Streamlit libraries
