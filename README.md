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
- `keypoint_detection/`: Keypoint detection module
  - `preprocess.py`:Script for importing images and extracting hand landmark data
  - `train.py`: Training script
- `test/`: Evaluation and comparison scripts
  - `Keypoint_Detection_Test.py`: Script for testing keypoint detection and gesture recognition
  - `Object_Detection_Test.py`: Script for testing object detection and gesture recognition
- `requirements.txt`: Project dependencies
- `README.md`: This file

---

## 📂 Dataset

Due to the large size of the dataset (~4GB), it is hosted externally.

🔗 **Download Dataset:**

- **Baidu Pan (百度网盘)**: [datasets-Comparative_Study_of_Object_Detection_and_Keypoint_Detection_for_Static_Chinese_Sign_Language_Recognition.zip](https://pan.baidu.com/s/1Q7XzqUVXfNJuVTt823LePw?pwd=na79)  

- **Google Drive**: [datasets-Comparative_Study_of_Object_Detection_and_Keypoint_Detection_for_Static_Chinese_Sign_Language_Recognition.zip](https://drive.google.com/file/d/1iQALh2GAQarPmaReNwFGdtSeIMWaixv0/view?usp=drive_link)

**Contents:**

- `images/`: Static Chinese sign language images
- `object-labels/`: Object detection labels
- `keypoint-labels/`: Keypoint detection labels

---

## 🚀 Getting Started

### Requirements
#### Keypoint Detection Requirements
- Python ≥ 3.11
- TensorFlow  ≥ 2.19
- OpenCV
- MediaPipe

#### Object Detection Requirements
- Python ≥ 3.11
- PyTorch ≥ 2.2
- OpenCV
- YOLO11

Install dependencies:

```bash
pip install -r requirements.txt
