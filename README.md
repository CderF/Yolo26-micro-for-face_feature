# YOLO26-Micro: Customized Lightweight Architecture for Micro-expression Recognition in Online Learning

>## **Languages:** [简体中文](README_zh.md) | [English](README.md)

## 📌 Motivation
### This project is a deep secondary development based on the experimental branch of Ultralytics YOLO26.

Standard YOLO architectures (based on P3-P4-P5) suffer from critical bottlenecks when tasked with "Micro-expression Recognition (MER) in online education":

Resolution Collapse: Micro-expressions involve extremely subtle movements (often < 10 pixels), which are physically erased in deep P5 layers (32x downsampling).

High-Frequency Noise: Complex backgrounds in home environments (furniture, screen reflections) easily drown out weak facial muscle activation signals.

YOLO26-Micro is designed with a core philosophy: "Discard P5, Integrate P2, and Fortify with Attention."

## 🛠️ Core Architectural Innovations
### 1. Discarding the "Blind" P5 Layer & Investing in P2 Ultra-Clear Layer
Severed the backbone path to P5 (20x20), capping the highest layer at P4 (40x40).

Extracted the P2 feature layer (160x160, only 4x downsampling) from the shallow backbone. This ensures that even a 4-pixel twitch at the corner of an eye occupies a full feature grid.

### 2. C2PSA as the "Breakwater" for P2
Integrated C2PSA (Channel-to-Pixel Spatial Attention) at the P2 layer before it enters the Neck.

This suppresses background grid activation and highlights facial contours, preventing subtle expression signals from being washed away by noise during subsequent feature fusion.

### 3. Reconstructing the Multi-Scale "Interchange" (Neck)
Reconfigured the traditional P5-P4-P3 FPN+PANet into a P4-P3-P2 cascaded fusion.

The Detect Head outputs are redirected to monitor: [Ultra-High Res (P2), High Res (P3), Medium Res (P4)].

### 4. Pushing the Physical Limits of Scale
To satisfy the underlying mathematical requirement of C2PSA (avoiding ZeroDivisionError when hidden compression is 0.5), the model scale was adjusted from Nano to Small: scales: s: [0.33, 0.50, 1024].

This ensures at least 1 attention head operates precisely while keeping total parameters strictly within the threshold for real-time edge inference.

## 📄 Configuration File:
Yolo26-micro-for-face_feature/ultralytics/cfg/models/26/yolo26-micro.yaml

## 🚀 Roadmap
### [ ] Refine the Attention Residual Mechanism.

### [ ] Modify the first Convolutional layer to retain more texture details.

### [ ] Rewrite the Classify class to replace the current Detect head for facial feature tasks.

### [ ] Implement alternative Loss Functions (e.g., Focal/Inner-IoU).

## ⚠️ Disclaimer
This project is entirely based on the Ultralytics open-source framework. The yolo26-micro script is NOT an official Ultralytics release.

This is a preliminary upload. If any formal legal notices or attributions are missing, please open an Issue. Thank you for your understanding!
