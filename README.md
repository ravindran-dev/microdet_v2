# MicroDet – Lightweight Drone-Based Object Detection

A lightweight, anchor-free object detection system built using MicroDet, optimized for aerial / drone imagery.
This project focuses on efficient person detection with minimal computational overhead, making it suitable for edge devices and UAV applications.

---

## Executive Summary

MicroDet delivers efficient, anchor-free person detection for drone imagery with a practical train-validate-infer pipeline.
The repository is designed for experimentation speed, stable training behavior, and deployment-friendly inference.

| Focus Area | Value |
| --- | --- |
| Detection Strategy | Anchor-free regression with DFL |
| Deployment Goal | Lightweight edge/UAV compatibility |
| Data Standard | COCO-style annotations |
| Training Stability | AMP + EMA + configurable assigner/loss |

## Quick Navigation

- [Features](#features)
- [Project Structure](#project-structure)
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Configuration Highlights](#configuration-highlights)
- [Tech Stack](#tech-stack)
- [Applications](#applications)
- [Future Work](#future-work)
- [Author - Ravindran S](#--author---ravindran-s)

## Quick Start Workflow

### 1) Install dependencies

```zsh
python -m pip install -r requirements.txt
```

### 2) Configure experiment

Update `microdet.toml` for dataset paths, optimizer settings, assigner radius, and loss options.

### 3) Train and validate

Use the training commands in the Training section to start from scratch or resume from a checkpoint.

### 4) Run inference

Use the inference command in the Inference section to generate visual outputs with NMS-applied detections.

### 5) Review outputs

Check model checkpoints in `runs/microdet_drone/weights/` and logs in `runs/microdet_drone/logs/`.

---

## Features

- Lightweight MicroDet architecture

- Anchor-free detection (DFL-based regression)

- COCO-format dataset support

- Mixed-precision (AMP) training

- EMA (Exponential Moving Average) weights

- End-to-end pipeline: Train → Validate → Infer

- Bounding box visualization with NMS

- Config-driven (.toml) model & training setup

## Project Structure
```yaml
microdet/
├── tmp/
│   ├── model/
│   │   ├── backbone/
│   │   ├── neck/
│   │   ├── detect/
│   │   ├── loss/
│   │   └── model_wrapper.py
│   ├── train/
│   │   ├── train.py
│   │   └── validate.py
│   ├── infer/
│   │   ├── run_infer.py
│   │   └── image.png
│   └── data/
│       ├── coco_dataset.py
│       └── collate.py
├── runs/
│   └── microdet_drone/
│       ├── weights/
│       │   ├── last.ckpt
│       │   └── best.ckpt
│       └── logs/
├── microdet.toml
├── requirements.txt
└── README.md
```
## Model Overview

MicroDet is a one-stage, anchor-free detector designed for speed and efficiency.

### Architecture

- Backbone: Lightweight CNN for feature extraction

- Neck: Multi-scale feature aggregation

- Head:

  - Classification branch (Quality Focal Loss)

  - Regression branch (Distribution Focal Loss – DFL)

### Feature Map Strides
```csharp
[8, 16, 32]
```
## Loss Functions
| Loss Type                   | Purpose                    |
|----------------------------|----------------------------|
| Quality Focal Loss (QFL)   | Classification confidence  |
| Distribution Focal Loss (DFL) | Bounding box regression |
| GIoU Loss                  | Box overlap accuracy       |

### Configured in microdet.toml:
```toml
[model.head.loss]
config = [2.0, 0.25, 1.0, 7, "giou"]
```
## Dataset

- COCO-style annotation format

- Single class: person

- Input resolution: 640×640

- Supports training & validation splits

### Example:
```toml
[data.train]
config = [
  "tmp/data/dataset/images",
  "tmp/data/dataset/result.json",
  [640, 640],
  true,
  {}
]
```
## Training
### Train from scratch
```zsh
python -m src.train.train \
  --config microdet.toml \
  --device cuda
```
### Resume training
```zsh
python -m src.train.train \
  --config microdet.toml \
  --device cuda \
  --resume runs/microdet_drone/weights/last.ckpt
```
## Evaluation

- Validation runs every val_interval epochs

- Metrics:

  - mAP
  
  - Confidence stability
  
  - Qualitative bounding box accuracy

## Inference

### Run inference on a test image:
```zsh
python tmp/infer/run_infer.py
```
### Output

- Bounding boxes drawn on original image

- Non-Maximum Suppression (NMS) applied

### Output saved to:
```zsh
tmp/infer/output.png
```
## Sample Output

✔ Detected persons from aerial view
✔ Bounding boxes after NMS
✔ Scaled correctly to original image resolution

Early training may show many low-confidence boxes; tuning assigner radius and confidence threshold improves results.

## Configuration Highlights
### Assigner (Important)
```toml
[model.head.assigner_cfg]
config = ["CenterAssigner", {
  "8"  = 5.0,
  "16" = 5.0,
  "32" = 5.0
}]
```
### Optimizer
```toml
[schedule.optimizer]
config = ["adamw", 0.0005, 0.05, true, true]
```
## Known Challenges

- Low confidence during early epochs

- Over-detection without NMS

- DFL decoding tensor contiguity issues

- Correct stride handling during inference

✔ All addressed through architectural tuning and post-processing.

## Tech Stack

- Language: Python

- Framework: PyTorch

- Vision: OpenCV

- Model: MicroDet

- Data Format: COCO

- Hardware: CUDA GPU

## Applications

- Drone surveillance

- Crowd monitoring

- Search & rescue

- Smart city analytics

- Edge AI deployments

## Future Work

- Multi-class detection

- Video inference & tracking

- Model quantization

- Edge deployment (Jetson / TPU)

- Knowledge distillation

##  Author - **Ravindran S** 


Developer • ML Enthusiast • Neovim Customizer • Linux Power User  

Hi! I'm **Ravindran S**, an engineering student passionate about:

-  Linux & System Engineering  
-  AIML (Artificial Intelligence & Machine Learning)  
-  Full-stack Web Development  
-  Hackathon-grade project development  





## 🔗 Connect With Me

You can reach me here:

###  **Socials**
<a href="www.linkedin.com/in/ravindran-s-982702327" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white">
</a>


<a href="https://github.com/ravindran-dev" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-111111?style=for-the-badge&logo=github&logoColor=white">
</a>


###  **Contact**
<a href="mailto:ravindrans.dev@gmail.com" target="_blank">
  <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white">
</a>
