# Insulator Defect Detection with YOLO

Comparative insulator defect detection using YOLOv8n, YOLOv11n, and YOLOv26n with image and video inference through a Streamlit app.

![App Interface](images/interface.png)

## Overview

This project benchmarks three Ultralytics YOLO models (v8n, v11n, v26n) on an insulator defect detection dataset.  
It focuses on both detection quality and practical deployment, with a Streamlit interface that supports image and video inference to make the differences between models visually clear.[cite:262][cite:333]

## Key Features

- Training and evaluation of YOLOv8n, YOLOv11n, and YOLOv26n on the same insulator dataset.[cite:262][cite:334]  
- Side‑by‑side visual comparison of detections on test images.  
- Streamlit app for interactive image upload and multi‑model video processing.  
- Simple, lightweight repository without bundled training dataset or full `runs` folder.

## Dataset

- Task: Insulator defect detection on overhead line insulators.  
- Format: YOLO‑style dataset with train/val/test splits and bounding box annotations.  
- Source: Publicly available insulator dataset (linked in the notebook / app configuration).  

> Note: The full dataset is not included in this repository to keep it lightweight.  
Paths and dataset settings are documented in the notebook and config files.

## Model Comparison

The project compares three models under identical training and evaluation settings, using metrics such as precision, recall, and mAP on the held‑out test set.[cite:262][cite:334]

![Metrics Summary](images/metrics.png)

YOLOv11n achieves the best overall precision and mAP in this setup, while YOLOv8n offers slightly higher recall and YOLOv26n performs worst among the three.[cite:262]

![Visual Comparison](images/comparison.png)

The visual comparison image shows typical detection outputs from each model, making the qualitative differences easy to see at a glance.

## Streamlit App

The Streamlit app exposes the trained models for interactive use:

- Upload an image and see detections from all three models side‑by‑side.  
- Upload a video and generate three processed videos, one per model, to compare behaviour over time.  

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

(Adjust the script path if your app file lives inside the `insulator_project` folder.)

## Repository Structure

- `YOLO.ipynb` – main notebook for dataset checks, training, and evaluation.  
- `insulator_project/` – app code, configuration, and supporting scripts.  
- `images/` – selected screenshots for the README (interface, metrics, visual comparison).  
- `requirements.txt` – Python dependencies for running the app and notebook.

## Future Work

- Add more detailed ablations (different image sizes, training epochs, or augmentation strategies).  
- Extend the app to support model selection and threshold tuning from the UI.  
- Explore exporting the best model to ONNX or TensorRT for faster edge deployment.

## Author

Abhishek Pawar