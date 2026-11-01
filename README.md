# 🚦 Automated Traffic Sign Recognition System for Autonomous Vehicles


An end-to-end computer vision and deep learning perception pipeline designed for Advanced Driver Assistance Systems (ADAS). The architecture normalizes environmental sensor degradations and classifies 43 distinct traffic sign categories in real time using a lightweight Convolutional Neural Network (CNN) optimized for edge compute.

---

## 📌 Problem Statement & Engineering Focus

Autonomous perception pipelines require deterministic, low-latency visual inference under safety-critical constraints. Standard classification models often fail in real-world driving environments due to:
* **Illumination shifts:** Heavy shadows, headlight glare, overexposure, and low-light nocturnal conditions.
* **Environmental noise:** Motion blur, weather occlusions, and off-axis camera angles.
* **Edge resource limits:** On-vehicle Electronic Control Units (ECUs) demand compact memory footprints and low computational overhead ($<10\text{ ms}$ inference).

This solution pairs an adaptive computer vision preprocessing pipeline with an 11-layer CNN ($378\text{K}$ parameters) that delivers **96.63% accuracy across 12,630 unseen benchmark test images** and supports zero-shot real-world generalization.

---

## 🏗️ End-to-End Pipeline Architecture

```
Raw Camera Feed (32×32×3 RGB)
           │
           ▼
[Stage 1: Illumination-Invariant Preprocessing]
  ├── Grayscale Conversion   ──► Drops 3 channels to 1 (66% memory reduction)
  ├── Histogram Equalization ──► Normalizes contrast across extreme lighting
  └── Min-Max Normalization  ──► Scales pixel values to [0, 1] for stable gradients
           │
           ▼
[Stage 2: Dynamic Data Augmentation]
  └── Stochastic Transformations: ±10° rotation, 10% translation, 20% zoom, 10% shear
           │
           ▼
[Stage 3: Deep Feature Extraction]
  ├── Block 1: 2× Conv2D (60 filters, 5×5) + MaxPool (2×2) ──► Low-level edges & contours
  ├── Block 2: 2× Conv2D (30 filters, 3×3) + MaxPool (2×2) ──► Geometric shape synthesis
  └── Regularization: Spatial Dropout (0.5)
           │
           ▼
[Stage 4: Dense Classifier & Decision Logic]
  ├── Dense (500 units, ReLU) + Dropout (0.5)
  └── Dense Softmax Layer (43 Output Classes)
           │
           ▼
[Stage 5: Validated Real-World Classification]
```

---

## 🔬 Computer Vision Preprocessing

Raw RGB pixel feeds introduce computational overhead and vulnerability to sensor exposure variations. A 3-step deterministic transformation standardizes every frame prior to inference:

1. **Grayscale Conversion (`cv2.cvtColor`):** Road signs are primarily identified by geometric layout and iconography. Reducing depth from 3 channels to 1 reduces tensor bandwidth by 66%.
2. **Contrast Enhancement via Histogram Equalization (`cv2.equalizeHist`):** Redistributes pixel intensities to recover edge boundaries in underexposed or backlit images.
3. **Statistical Normalization:** Normalizes input range to $[0.0, 1.0]$, preventing exploding gradients and accelerating Adam convergence.

---

## 🧠 Neural Network Architecture

The network maximizes feature representation while maintaining a minimal memory footprint ($378,023$ trainable parameters / $1.44\text{ MB}$), enabling on-chip cache execution on embedded automotive hardware.

| Layer | Type | Output Shape | Kernel / Stride | Activation | Parameters | Function |
|---|---|---|---|---|---|---|
| **0** | `InputLayer` | `(32, 32, 1)` | — | — | 0 | Single-channel normalized tensor |
| **1** | `Conv2D` | `(28, 28, 60)` | $5 \times 5$ | ReLU | 1,560 | Primary boundary/edge extraction |
| **2** | `Conv2D` | `(24, 24, 60)` | $5 \times 5$ | ReLU | 90,060 | Local feature combinations |
| **3** | `MaxPooling2D`| `(12, 12, 60)` | $2 \times 2$ | — | 0 | Spatial subsampling & translation invariance |
| **4** | `Conv2D` | `(10, 10, 30)` | $3 \times 3$ | ReLU | 16,230 | Intermediate shape detection |
| **5** | `Conv2D` | `(8, 8, 30)` | $3 \times 3$ | ReLU | 8,130 | Complex pattern & symbol synthesis |
| **6** | `MaxPooling2D`| `(4, 4, 30)` | $2 \times 2$ | — | 0 | Subsampling & dimension reduction |
| **7** | `Dropout` | `(4, 4, 30)` | Rate = 0.5 | — | 0 | Spatial co-adaptation prevention |
| **8** | `Flatten` | `(480)` | — | — | 0 | Multi-dimensional tensor unrolling |
| **9** | `Dense` | `(500)` | — | ReLU | 240,500 | High-dimensional classification manifold |
| **10**| `Dropout` | `(500)` | Rate = 0.5 | — | 0 | Dense feature regularizer |
| **11**| `Dense` | `(43)` | — | Softmax | 21,543 | Probability distribution over 43 classes |

* **Loss Function:** Categorical Cross-Entropy
* **Optimizer:** Adam ($\alpha = 0.001$)
* **Batch Size:** 50 samples/batch ($696\text{ steps/epoch}$)

---

## 📊 Benchmark Dataset & Empirical Performance

Trained and evaluated on the **German Traffic Sign Recognition Benchmark (GTSRB)**:

| Dataset Partition | Samples | Input Dimensions | Modality |
|---|---|---|---|
| **Training Split** | 34,799 | $32 \times 32$ | Preprocessed Single-Channel |
| **Validation Split** | 4,410 | $32 \times 32$ | Preprocessed Single-Channel |
| **Held-Out Test Set** | 12,630 | $32 \times 32$ | Preprocessed Single-Channel |
| **Total Evaluation** | **51,839** | — | **43 Classes (IDs 0–42)** |

### Performance Metrics

* **Held-Out Test Accuracy:** **96.63%** across 12,630 unseen images.
* **Test Categorical Loss:** **0.1224**.
* **Validation Accuracy:** Peak **98.71%** (Validation Loss: **0.0479**).
* **Inference Latency:** $\sim 6\text{ ms}$ per sample on CPU ($\approx 160\text{ FPS}$), exceeding real-time ADAS constraints ($>30\text{ FPS}$).

---

## 🧪 Zero-Shot Real-World Image Verification

The pipeline was stress-tested against an uncurated external stock photograph (*"Turn left ahead"* sign) to evaluate out-of-domain transferability:
1. **ROI Extraction:** Cropped to isolate the sign from watermark noise and background clutter.
2. **Preprocessing:** Rescaled to $32 \times 32$, converted to grayscale, contrast-equalized, and normalized.
3. **Outcome:** Correctly classified as **Class 34 (*Turn left ahead*)** with high model confidence ($>90\%$).

---

## 🚀 Quickstart & Inference

### 1. Interactive Google Colab
Run the complete training, evaluation, and test suite directly in the cloud:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NOomop4358GWzHPwfR7P4RsiGE_tK5Qm?usp=sharing)

### 2. Local Setup

```bash
# Clone repository
git clone [https://github.com/Ashish3692003/Automated-Traffic-Sign-CNN.git](https://github.com/Ashish3692003/Automated-Traffic-Sign-CNN.git)
cd Automated-Traffic-Sign-CNN

# Create virtual environment and install requirements
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install tensorflow opencv-python numpy pandas matplotlib pillow requests
```

### 3. Inference Script

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    normalized = equalized / 255.0
    return normalized.reshape(1, 32, 32, 1)

# Usage
# model = load_model('traffic_sign_cnn.h5')
# processed = preprocess_image(cv2.imread('road_sign.jpg'))
# class_id = np.argmax(model.predict(processed))
```

---

## 🔮 Production Roadmap

* [ ] **Upstream Object Detection:** Integrate YOLOv8/v9 to detect and crop bounding-box ROIs from raw $1080\text{p}/4\text{K}$ automotive video streams.
* [ ] **Edge Hardware Quantization:** Convert model to **TensorFlow Lite (TFLite)** and **TensorRT** with INT8 post-training quantization for deployment on embedded NPUs, NVIDIA Jetson, and automotive FPGA accelerators.
* [ ] **Temporal Consensus:** Implement multi-frame majority voting across successive video frames to eliminate single-frame transient false positives.
