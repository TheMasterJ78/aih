import os
import numpy as np
from PIL import Image
import pandas as pd

base_dir = "C:/Users/sauha/.gemini/antigravity-ide/scratch/skin_disease_project/dataset"
img_dir = os.path.join(base_dir, "images")
os.makedirs(img_dir, exist_ok=True)

classes = ['Monkeypox', 'Measles', 'Chickenpox']
data = []

np.random.seed(42)

print("Generating synthetic images and expanded clinical data...")
for class_idx, class_name in enumerate(classes):
    for i in range(100):
        # Generate a synthetic image
        img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        if class_idx == 0: img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + 50, 0, 255) # Reddish
        elif class_idx == 1: img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] + 50, 0, 255) # Greenish
        else: img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] + 50, 0, 255) # Blueish
        
        img = Image.fromarray(img_arr)
        img_name = f"{class_name}_{i}.jpg"
        img_path = os.path.join(img_dir, img_name)
        img.save(img_path)
        
        # Generate EXTENDED synthetic clinical data
        temp = np.random.uniform(98.6, 101.0) if class_idx != 0 else np.random.uniform(100.0, 103.0)
        lymph = 1 if class_idx == 0 and np.random.rand() > 0.2 else (1 if np.random.rand() > 0.8 else 0)
        days = np.random.randint(1, 14)
        
        # NEW MEDICAL FEATURES
        itchiness = np.random.randint(0, 11) # 0 to 10 scale
        headache = 1 if np.random.rand() > 0.4 else 0
        fatigue = 1 if np.random.rand() > 0.3 else 0
        # Travel history is a big indicator for Mpox
        travel = 1 if class_idx == 0 and np.random.rand() > 0.1 else (1 if np.random.rand() > 0.9 else 0) 
        
        data.append({
            'image_id': img_name,
            'temperature': temp,
            'swollen_lymph_nodes': lymph,
            'days_since_rash': days,
            'itchiness_level': itchiness,
            'headache': headache,
            'severe_fatigue': fatigue,
            'recent_travel': travel,
            'label': class_idx,
            'disease_name': class_name
        })

df = pd.DataFrame(data)
df.to_csv(os.path.join(base_dir, "clinical_data.csv"), index=False)
print(f"Dataset generated successfully! {len(df)} records created with 7 medical features in {base_dir}")
