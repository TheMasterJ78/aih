import os
import shutil
import pandas as pd
import numpy as np

base_dir = "C:/Users/sauha/.gemini/antigravity-ide/scratch/skin_disease_project/dataset"
img_dir = os.path.join(base_dir, "images")
os.makedirs(img_dir, exist_ok=True)

# Clear existing images
for f in os.listdir(img_dir):
    os.remove(os.path.join(img_dir, f))

data = []
np.random.seed(42)

def add_metadata(new_name, class_idx, class_name):
    temp = np.random.uniform(98.6, 101.0) if class_idx != 0 else np.random.uniform(100.0, 103.0)
    lymph = 1 if class_idx == 0 and np.random.rand() > 0.2 else (1 if np.random.rand() > 0.8 else 0)
    days = np.random.randint(1, 14)
    itchiness = np.random.randint(0, 11)
    headache = 1 if np.random.rand() > 0.4 else 0
    fatigue = 1 if np.random.rand() > 0.3 else 0
    travel = 1 if class_idx == 0 and np.random.rand() > 0.1 else (1 if np.random.rand() > 0.9 else 0)
    
    data.append({
        'image_id': new_name,
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

copied_count = 0
for root, dirs, files in os.walk(base_dir):
    # Skip the output directory
    if os.path.abspath(root) == os.path.abspath(img_dir):
        continue
        
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, f)
            path_lower = full_path.lower()
            
            # Simple heuristic based on folder or file names
            if "measle" in path_lower:
                class_idx, class_name = 1, 'Measles'
            elif "chicken" in path_lower or "cpox" in path_lower:
                class_idx, class_name = 2, 'Chickenpox'
            elif "monkey" in path_lower or "mpox" in path_lower:
                class_idx, class_name = 0, 'Monkeypox'
            else:
                # To balance the dataset slightly if the file name is generic
                class_idx, class_name = 1, 'Measles' if copied_count % 2 == 0 else (2, 'Chickenpox')
                class_name = class_name if isinstance(class_name, str) else class_name[1]
                class_idx = class_idx if isinstance(class_idx, int) else class_idx[0]
                
            new_name = f"{class_name}_{copied_count}_{f}"
            dst = os.path.join(img_dir, new_name)
            try:
                shutil.copy(full_path, dst)
                add_metadata(new_name, class_idx, class_name)
                copied_count += 1
            except Exception:
                pass

df = pd.DataFrame(data)
df.to_csv(os.path.join(base_dir, "clinical_data.csv"), index=False)
print(f"Aggregated {copied_count} original RAW images!")
