# Infectious Skin Disease AI - Project Overview

This document provides a comprehensive overview of the architecture, machine learning models, data pipelines, and technical stack used to build the **BMC AI-Assisted Field Surveillance** application.

## 1. Project Goal & Overview
The project is an AI-assisted field surveillance portal designed for the Brihanmumbai Municipal Corporation (BMC). Its primary goal is to provide a door-to-door clinical intake system capable of diagnosing infectious skin diseases—specifically **Monkeypox**, **Measles**, and **Chickenpox**—using a combination of patient images (skin lesions) and clinical metadata (symptoms).

## 2. Technical Stack & Frontend
- **Framework**: Python with [Streamlit](https://streamlit.io/) (`app.py`).
- **Styling**: Custom CSS injected into Streamlit for a professional, medical-grade User Interface (UI).
- **Report Generation**: Uses `fpdf2` to dynamically generate downloadable clinical PDF reports for patients based on AI assessments.
- **Deep Learning Framework**: TensorFlow & Keras.

## 3. Machine Learning Architecture
The core of the application is a **Multimodal Fusion Neural Network** that processes both visual and tabular data simultaneously to make a prediction. 

### A. Image Branch (Transfer Learning)
- **Base Model**: MobileNetV2 (pretrained on ImageNet).
- **Input Size**: 224x224 RGB images.
- **Setup**: The base layers of MobileNetV2 are frozen. The output is passed through a `GlobalAveragePooling2D` layer and a `Dense` layer (128 units, ReLU activation) to extract visual features.

### B. Tabular Branch (Clinical Data)
- **Inputs**: 7 clinical data points (Body Temperature, Swollen Lymph Nodes, Days Since Rash, Itchiness Level, Headache, Severe Fatigue, Recent Travel).
- **Setup**: Processed through two `Dense` layers (32 units and 16 units, both with ReLU activation).

### C. Fusion & Classification Layer
- The visual features (from the image branch) and clinical features (from the tabular branch) are concatenated.
- A final `Dense` layer with 3 units and `softmax` activation outputs the probability for the three classes: **Monkeypox (0)**, **Measles (1)**, and **Chickenpox (2)**.
- The model is compiled using the `Adam` optimizer and `sparse_categorical_crossentropy` loss.

### D. Cloud Vision AI (Fallback)
The application includes a robust fallback mechanism. If the user selects "Cloud Vision API" or if local processing is insufficient, the system routes the image and symptom text to **Google's Gemini 1.5 Pro** via the Generative AI API to perform zero-shot diagnostic cross-validation.

## 4. Dataset & Data Pipeline
The data pipeline is designed to combine real images with synthetically generated, medically accurate metadata for training.

- **Data Source**: Kaggle's "Monkeypox Skin Lesion Dataset" (and similar open-source datasets).
- **Data Preparation (`prepare_kaggle_data.py`)**: 
  - Extracts and organizes images into respective classes based on file/folder names.
  - **Synthetic Metadata Generation**: Generates synthetic clinical records (e.g., higher temperature and swollen lymph nodes for Monkeypox; high itchiness for Chickenpox) using randomized algorithms seeded with clinical heuristics.
  - Saves the tabular data to `clinical_data.csv`.

## 5. Training Pipeline (`train_model.py`)
- **Preprocessing**: Images are resized to 224x224 and normalized (pixel values divided by 255.0). Tabular data is extracted directly from the generated CSV.
- **Training**: The model is trained using `model.fit([X_img, X_tab], y, epochs=5, batch_size=16)`. (Note: This is a prototype configuration; epochs and augmentation should be increased for production).
- **Export**: The trained multimodal model is exported and saved as an HDF5 file (`multimodal_model.h5`), which is then loaded directly by the Streamlit frontend.
