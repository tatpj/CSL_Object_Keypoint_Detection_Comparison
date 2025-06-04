# Comparative Study of Object Detection and Keypoint Detection for Static Chinese Sign Language Recognition

This repository contains the source code and dataset related to our paper:

**"Comparative Study of Object Detection and Keypoint Detection for Static Chinese Sign Language Recognition"**  
📄 Presented at ICEICT 2025

## 📦 Contents

- `models/`: Pre-trained model weights
  - `object-detection-best.pt`: Best object detection model
  - `keypoint-detection-best.pt`: Best keypoint detection model
- `object_detection/`: Object detection module
  - `csl.yaml`: Dataset configuration file
  - `train.py`: Training script
  - `weights/`: Model weights
- `keypoint_detection/`: Keypoint detection module
  - `preprocess.py`:Script for importing images and extracting hand landmark data
  - `train.py`: Training script
  - `weights/`: Model weights
- `requirements.txt`: Project dependencies
- `README.md`: This file

---

## 📂 Dataset

Due to the large size of the dataset (~4GB), it is hosted externally.

🔗 **Download Dataset:**

- **Baidu Pan (百度网盘)**: [datasets-Comparative_Study_of_Object_Detection_and_Keypoint_Detection_for_Static_Chinese_Sign_Language_Recognition.zip](https://pan.baidu.com/s/1Unr6m97wjuNBZnSIftOviQ?pwd=417s)  
  Extraction code: `417s`

- **Google Drive**: [datasets-Comparative_Study_of_Object_Detection_and_Keypoint_Detection_for_Static_Chinese_Sign_Language_Recognition.zip](https://drive.google.com/file/d/1aSrWut3HGIgrTgvLb0EDoFmpwH2avpdI/view?usp=sharing)

**Contents:**

- `images/`: Static Chinese sign language images
- `object-labels/`: Object detection labels
- `keypoint-labels/`: Keypoint detection labels

---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.20
- OpenCV
- YOLO11

Install dependencies:

```bash
pip install -r requirements.txt
