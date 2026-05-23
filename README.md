# 🚦 Automated Traffic Sign Recognition System using Convolutional Neural Networks (CNN)

> **Developed as part of the Deep Learning Case Study at GITAM University**

---

## 📋 Project Overview

The **Automated Traffic Sign Recognition System** is a Deep Learning model capable of classifying 43 distinct categories of traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The system is implemented in **Python** using the **TensorFlow/Keras** framework on **Google Colab**, demonstrating high accuracy and generalization in image classification for autonomous vehicle safety.

---

## 🎯 Objective

Traffic Sign Recognition (TSR) is a critical component of Advanced Driver Assistance Systems (ADAS) and autonomous vehicles. The inability of a vehicle to recognize and obey traffic regulations can lead to catastrophic accidents. This project addresses challenges such as varying lighting conditions and occlusions by employing Convolutional Neural Networks (CNNs) to automatically learn hierarchical feature representations from traffic sign images — moving beyond manual feature extraction.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Platform** | Google Colab |
| **Dataset** | GTSRB (51,839 images) |
| **Image Processing** | OpenCV (cv2) |
| **Libraries** | NumPy, Pandas, Matplotlib, Scikit-learn, Pickle |

---

## 📊 Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)**, sourced from the Institut für Neuroinformatik, Ruhr-Universität Bochum.

| Split | Number of Images |
|-------|-----------------|
| **Training Set** | ~34,799 |
| **Validation Set** | ~4,410 |
| **Test Set** | ~12,630 |
| **Total** | **~51,839** |

- **Number of Classes:** 43 (IDs 0–42)
- **Image Size:** 32 × 32 pixels
- **Initial Format:** RGB (3 color channels)
- **Data Source:** Pickled files (`train.p`, `valid.p`, `test.p`) obtained by cloning a Git repository

---

## 🔧 Methodology

The system follows an end-to-end pipeline with the following stages:

### 1. Data Loading
Images and labels are loaded from pickled files (`train.p`, `valid.p`, `test.p`) containing `X_train`, `X_val`, `X_test` (images) and `y_train`, `y_val`, `y_test` (labels).

### 2. Data Visualization
- **Sign Name Mapping:** A `signnames.csv` file is loaded to map class IDs (0–42) to human-readable traffic sign names.
- **Sample Display:** Random images from each of the 43 classes are displayed to provide a visual overview.
- **Distribution Analysis:** A bar chart is generated to show the distribution of samples across all 43 classes, highlighting potential class imbalances.

### 3. Data Preprocessing
- **Grayscale Conversion:** All images are converted from RGB to grayscale using `cv2.cvtColor`. This reduces dimensionality and is effective where shape/texture matters more than color.
- **Histogram Equalization:** `cv2.equalizeHist` is applied to enhance contrast, making features more discernible.
- **Normalization:** Pixel values are scaled to the range [0, 1] by dividing by 255, stabilizing training and speeding up convergence.
- **Reshaping:** Images are reshaped to `(32, 32, 1)` to match the CNN input requirements.

### 4. Data Augmentation
To improve model generalization and prevent overfitting, `ImageDataGenerator` is applied with:
- Random width/height shifts
- Zooms
- Shears
- Rotations

### 5. Label Encoding
Categorical labels are converted to one-hot encoded vectors using `to_categorical`.

---

## 🏗️ CNN Architecture

The Sequential CNN architecture consists of **378,023 total trainable parameters**.

| Layer | Type | Filters / Units | Kernel Size | Activation | Notes |
|-------|------|----------------|-------------|------------|-------|
| 1 | Conv2D | 60 | 5×5 | ReLU | Input shape: (32, 32, 1) |
| 2 | Conv2D | 60 | 5×5 | ReLU | — |
| 3 | MaxPooling2D | — | 2×2 | — | — |
| 4 | Conv2D | 30 | 3×3 | ReLU | — |
| 5 | Conv2D | 30 | 3×3 | ReLU | — |
| 6 | MaxPooling2D | — | 2×2 | — | — |
| 7 | Dropout | — | — | — | Rate: 0.5 |
| 8 | Flatten | — | — | — | — |
| 9 | Dense | 500 | — | ReLU | Hidden fully connected layer |
| 10 | Dropout | — | — | — | Rate: 0.5 |
| 11 | Dense | 43 | — | Softmax | Output layer (43 classes) |

**Optimizer:** Adam (Learning Rate = 0.001)
**Loss Function:** Categorical Cross-Entropy
**Metrics:** Accuracy

---

## 💻 Code Implementation

### Key Imports
```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Model Definition
```python
def neural_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

---

## 📈 Results

### Training Performance
The model was trained for **10 epochs** with validation performed at each epoch.

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | **97.96%** |
| **Final Validation Loss** | **0.0807** |
| **Number of Epochs** | 10 |
| **Total Trainable Parameters** | **378,023** |

### Test Set Evaluation
The model was evaluated on the held-out test set of ~12,630 images.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **95.07%** |
| **Test Loss** | **0.1731** |

### Custom Image Prediction
The model was tested on a real-world image (downloaded from URL), which was preprocessed and fed into the trained model. It correctly identified the sign as **"Turn left ahead" (Class ID: 34)**.

---

## 🏁 Conclusion

The study confirms that Deep Learning architectures, particularly Convolutional Neural Networks, are highly effective for computer vision tasks in automotive safety. The CNN model successfully generalizes to unseen test data, demonstrating robustness through hierarchical feature extraction, image augmentation, and dropout regularization. This project provides a robust, reproducible pathway for real-world traffic sign recognition in autonomous driving systems.

---

## 📚 References

1. Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). *The German Traffic Sign Recognition Benchmark*. Institut für Neuroinformatik, Ruhr-Universität Bochum, Germany.
2. TensorFlow Documentation. (2024). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems*. https://www.tensorflow.org
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---
## 📚 Deep learning Case Study
This project was developed as part of the **Deep Learning** case study at **GITAM University, Visakhapatnam**. The case study required implementing a complete computer vision pipeline for real-time object detection, tracking, and counting on video streams.
**Google Colab Notebook:** [Open on Colab](https://colab.research.google.com/drive/1NOomop4358GWzHPwfR7P4RsiGE_tK5Qm?usp=sharing)

<p align="center">
Made with ❤️ by <strong>V. Ashish Nadh</strong> | <a href="https://github.com/Ashish3692003">View my GitHub</a>
</p>
