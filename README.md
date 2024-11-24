# RetiScan: Automated Diagnosis of Retinopathy of Prematurity (ROP)

## Overview
RetiScan is a deep learning-based system designed to assist in the diagnosis and staging of Retinopathy of Prematurity (ROP). The project leverages computer vision techniques and state-of-the-art neural networks to analyze retinal images, enabling automated detection and classification of ROP stages. The goal is to provide a reliable tool to support ophthalmologists in early diagnosis and treatment, potentially reducing the risk of blindness in premature infants.

---

## Features
- **Binary Classification:** Determines if an image shows signs of ROP (ROP vs. No ROP).
- **Stage Classification:** Identifies the severity of ROP into three stages:
  - Stage 1
  - Stage 2
  - Stage 3
- **Segmentation:** Employs U-Net models to generate masks for:
  - Blood vessels
  - Ridge structures
- **Performance Metrics:**
  - **Binary Classification Model (ResNet50):**
    - Accuracy: 98.40%
    - Precision: 98.59%
    - Recall: 97.22%
    - F1 Score: 97.90%
  - **Stage Classification Model (ResNet50):**
    - Accuracy: 92.80%
    - Precision: 92.94%
    - Recall: 92.89%
    - F1 Score: 92.74%
  - **Segmentation Model (U-Net):**
    - IoU (Blood vessels): 0.7532
    - IoU (Ridges): 0.7110

---

## Data
- **Dataset Composition:**
  - 400 high-quality images per ROP stage for training.
  - 80-20 train-test split.
  - Additional 2000 labeled images for binary classification training.
- **Image Preprocessing:**
  - Adaptive sigmoid enhancement for contrast improvement.
  - Separate color-channel mapping for segmentation outputs (blood vessels and ridges).

---

## Model Architecture
1. **Segmentation:**
   - **Model:** U-Net
   - **Purpose:** Generate masks for blood vessels and ridge structures.
2. **Binary Classification:**
   - **Model:** Fine-tuned ResNet50
   - **Purpose:** Classify images as ROP or No ROP.
3. **Stage Classification:**
   - **Model:** Fine-tuned ResNet50
   - **Purpose:** Classify ROP severity into predefined stages.

---

## Training Details
- **Hardware:** RTX 4060 (8GB VRAM)
- **Optimizer:** Adaptive learning rate scheduler in PyTorch.
- **Metrics:** Accuracy, Precision, Recall, F1 Score, IoU.

