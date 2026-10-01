# Skin Disease Dataset Acquisition Guide

To train our new MobileNetV2 model properly, we need high-quality, real medical data. Follow these steps to set up Kaggle and download an open skin disease dataset.

## 1. Get Kaggle API Credentials
1. Create an account or log in at [Kaggle](https://www.kaggle.com).
2. Go to your Account Settings (click your profile picture > Settings).
3. Scroll down to the **API** section and click **Create New API Token**.
4. This will download a file named `kaggle.json`.

## 2. Configure Kaggle on Windows
1. Open PowerShell or Command Prompt.
2. Create a `.kaggle` directory in your user folder:
   ```powershell
   mkdir C:\Users\%USERNAME%\.kaggle
   ```
3. Move the downloaded `kaggle.json` into this new directory.

## 3. Install Kaggle Library
Install the official Kaggle python library using pip:
```powershell
pip install kaggle
```

## 4. Download the Dataset
For this prototype, we recommend downloading a curated infectious disease dataset. Since exact Kaggle datasets shift over time, search Kaggle for "Monkeypox Skin Lesion Dataset" (like `nafin59/monkeypox-skin-lesion-dataset`).

Run this command in the project root (`c:\Users\sauha\.gemini\antigravity-ide\scratch\skin_disease_project\dataset`):
```powershell
kaggle datasets download -d nafin59/monkeypox-skin-lesion-dataset --unzip
```

## 5. Organize the Data
Ensure your downloaded images are placed inside `dataset/images/` and update your `clinical_data.csv` to map the new image names to the correct disease label (`0` for Monkeypox, `1` for Measles, `2` for Chickenpox).
