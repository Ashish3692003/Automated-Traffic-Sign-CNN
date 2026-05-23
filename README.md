# 🚦 Automated Traffic Sign Recognition System using Convolutional Neural Networks (CNN)

> **Deep Learning Application Case Study for Autonomous Vehicles**

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

- **Name:** German Traffic Sign Recognition Benchmark (GTSRB)
- **Source:** Institut für Neuroinformatik, Ruhr-Universität Bochum
- **Total Images:** 51,839
- **Image Size:** 32 × 32 pixels
- **Number of Classes:** 43
- **Data Splits:** Training, Validation, and Testing sets (pickled format: `train.p`, `valid.p`, `test.p`)

---

## 🔧 Methodology

### 1. Data Loading
Images are loaded from pickled files containing pre-organized training, validation, and test sets.

### 2. Data Preprocessing
- **Grayscale Conversion:** RGB images are converted to grayscale to reduce computational complexity.
- **Histogram Equalization:** Enhances contrast and improves feature visibility under varying lighting conditions.
- **Normalization:** Pixel values are scaled to the range [0, 1] by dividing by 255.
- **Reshaping:** Images are reshaped to `(32, 32, 1)` to match the CNN input requirements.

### 3. Data Augmentation
To improve model generalization and prevent overfitting, `ImageDataGenerator` is applied with:
- Random shifts
- Zooms
- Shears
- Rotations

### 4. Label Encoding
Categorical labels are converted to one-hot encoded vectors using `to_categorical`.

---

## 🏗️ CNN Architecture

The model follows a standard convolutional neural network architecture optimized for image classification:

| Layer | Type | Filters / Units | Kernel Size | Activation |
|-------|------|----------------|-------------|------------|
| 1 | Conv2D | 60 | 5×5 | ReLU |
| 2 | Conv2D | 60 | 5×5 | ReLU |
| 3 | MaxPooling2D | — | 2×2 | — |
| 4 | Conv2D | 30 | 3×3 | ReLU |
| 5 | Conv2D | 30 | 3×3 | ReLU |
| 6 | MaxPooling2D | — | 2×2 | — |
| 7 | Dropout | — | — | Rate: 0.5 |
| 8 | Flatten | — | — | — |
| 9 | Dense | 500 | — | ReLU |
| 10 | Dropout | — | — | Rate: 0.5 |
| 11 | Dense | 43 | — | Softmax |

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

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~97.96% (after 10 epochs) |
| **Test Set Accuracy** | **95.07%** |
| **Number of Epochs** | 10 |
| **Number of Classes** | 43 |

The model successfully demonstrated its ability to generalize to unseen test data. Custom prediction tests confirmed accurate classification — for example, a test image was correctly identified as **"Turn left ahead" (ID 34)**.

---

## 🏁 Conclusion

The study validates that Deep Learning architectures, particularly Convolutional Neural Networks, are highly effective for computer vision tasks in automotive safety. The CNN model effectively classifies traffic signs by extracting spatial features through convolutional layers and maintains robustness against variations via image augmentation and dropout regularization. This project provides a robust pathway for real-world traffic sign recognition in autonomous driving systems.

---

## 🔗 Links

- **Google Colab Notebook:** [Open on Colab](https://colab.research.google.com/drive/1NOomop4358GWzHPwfR7P4RsiGE_tK5Qm?usp=sharing)
- **GTSRB Dataset:** [Official Source](https://hci.iwr.uni-heidelberg.de/content/bosch-traffic-sign-dataset)

---

## 📚 References

1. Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011). The German Traffic Sign Recognition Benchmark. *Institut für Neuroinformatik, Ruhr-Universität Bochum, Germany.*
2. TensorFlow Documentation. (2024). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.*
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

---

> **Built with ❤️ using Python, TensorFlow & Keras**
