# Real Time Face Mask Detection

## 🎥 Demo
[![Demo Video](Demo/RealTime.gif)](Demo/RealTime.mp4)

This project implements a computer vision system that detects faces in images and classifies them into three categories: **with_mask**, **without_mask**, and **mask_incorrect**. Using transfer learning with MobileNetV2, the model achieves real-time face mask detection capabilities.

## 🚀 Key Features

- **Object Detection**: Extracts face regions from images using bounding box annotations
- **Multi-class Classification**: Classifies faces into three mask-wearing categories
- **Deep Learning**: Utilizes MobileNetV2 with custom classification head
- **Data Augmentation**: Implements image transformations to improve model robustness
- **Performance Evaluation**: Comprehensive metrics including confusion matrix and classification reports

## 🛠️ Technical Implementation

### Dataset
- **Source**: Face Mask Detection dataset from Kaggle
- **Format**: Pascal VOC XML annotations with corresponding images
- **Classes**: 
  - `with_mask` (correctly worn mask)
  - `without_mask` (no mask)
  - `mask_weared_incorrect` (improperly worn mask)

### Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Head**: Average Pooling → Flatten → Dense(128) → Dropout → Output(3)
- **Input Size**: 128×128×3 RGB images
- **Optimizer**: Adam with learning rate 1e-4

### Training
- **Epochs**: 10
- **Batch Size**: 32
- **Augmentation**: Rotation, zoom, shifts, shear, and horizontal flip

## ⭐ Don't forget to star this repository if you find it helpful!
