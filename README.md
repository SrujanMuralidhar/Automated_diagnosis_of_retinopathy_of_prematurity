# 🏥 Automated Diagnosis of Retinopathy of Prematurity (ROP)  

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)  
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange?logo=tensorflow)  
![License](https://img.shields.io/badge/License-MIT-green)  

## 📌 Overview  
This project automates **Retinopathy of Prematurity (ROP) detection and staging** using deep learning techniques.  
It employs **U-Net** for image segmentation and **ResNet50** for classification, achieving:  

✅ **98.40% accuracy** in ROP detection  
✅ **92.80% accuracy** in ROP stage classification  

## ✨ Features  
🚀 **Blood Vessel & Ridge Segmentation** using U-Net  
🔬 **Adaptive Image Enhancement** with Sigmoid Contrast Adjustment  
🤖 **Deep Learning Classification** via Fine-Tuned ResNet50  
📊 **High-Accuracy Screening Tool** for Real-World Clinical Use  

---

## 🗂 Dataset  
📍 **Source**: Narayana Nethralaya, Bengaluru, India 🇮🇳  
📸 **Size**: 4,185 patient records with **high-resolution fundus images**  
📝 **Annotations**: ROP Stage, Zone, Plus Disease, and Treatment Decisions  

---

## 🔬 Methodology  
### 🏗 Preprocessing  
- Resizing images to **512×512 pixels**  
- Removing **non-temporal images** using **U-Net-based optic disc segmentation**  

### 🔍 Feature Segmentation  
- **🩸 Blood Vessel Segmentation** using U-Net on original images  
- **🌀 Ridge Segmentation** using **Gabor-filter-enhanced U-Net**  

### 🔄 Classification  
- **📌 ROP Detection**: ResNet50 classifies **ROP vs. No ROP**  
- **📌 Stage-wise Classification**: Further classification into **Stage 1, 2, and 3** if ROP is detected  

---

## 📈 Performance  

### 🏥 ROP vs. No ROP Classification  

| Model            | 🎯 Accuracy | 🎯 Precision | 🔍 Recall | 📊 F1 Score |
|-----------------|------------|-------------|-----------|------------|
| **⚡ ResNet50**  | **98.40%**  | **98.59%**  | **97.22%** | **97.90%** |
| RegNet         | 94.68%     | 93.66%     | 92.36%    | 93.01%     |
| EfficientNetV2S | 95.21%     | 95.99%     | 94.68%    | 95.17%     |
| AlexNet        | 91.22%     | 91.31%     | 90.45%    | 90.82%     |
| GoogleNet      | 94.41%     | 94.16%     | 94.04%    | 94.10%     |

### 🔍 Stage-wise ROP Classification  

| Model            | 🎯 Accuracy | 🎯 Precision | 🔍 Recall | 📊 F1 Score |
|-----------------|------------|-------------|-----------|------------|
| **⚡ ResNet50**  | **92.80%**  | **92.94%**  | **92.89%** | **92.74%** |
| RegNet         | 89.83%     | 89.82%     | 89.82%    | 89.83%     |
| EfficientNetV2S | 89.41%     | 89.47%     | 89.98%    | 89.52%     |
| AlexNet        | 83.47%     | 83.55%     | 83.46%    | 83.50%     |
| GoogleNet      | 84.75%     | 84.91%     | 84.70%    | 84.66%     |

---

## ⚙️ Installation & Requirements  
### 📦 Dependencies  
🔹 Python 3.x  
🔹 TensorFlow / PyTorch  
🔹 OpenCV, NumPy, Pandas, Matplotlib  
🔹 Scikit-learn  

