import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2

base_dir = "C:/Users/sauha/.gemini/antigravity-ide/scratch/skin_disease_project/dataset"
df = pd.read_csv(os.path.join(base_dir, "clinical_data.csv"))

print("Loading images (224x224) and tabular data...")
images = []
for img_name in df['image_id']:
    img_path = os.path.join(base_dir, "images", img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    images.append(img_arr)

X_img = np.array(images)
X_tab = df[['temperature', 'swollen_lymph_nodes', 'days_since_rash', 'itchiness_level', 'headache', 'severe_fatigue', 'recent_travel']].values
y = df['label'].values

print("Building Multimodal Fusion Model (MobileNetV2 + Tabular)...")
# Branch 1: Image Input via Transfer Learning
img_input = Input(shape=(224, 224, 3), name='image_input')
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=img_input)
base_model.trainable = False # Freeze the base model for transfer learning

x1 = base_model.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dense(128, activation='relu')(x1)

# Branch 2: Tabular (Clinical) Input
tab_input = Input(shape=(7,), name='tabular_input')
x2 = Dense(32, activation='relu')(tab_input)
x2 = Dense(16, activation='relu')(x2)

# Fusion Layer
merged = Concatenate()([x1, x2])
output = Dense(3, activation='softmax', name='output')(merged)

model = Model(inputs=[img_input, tab_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
# Note: For the actual dataset, you may want to increase epochs and add data augmentation.
model.fit([X_img, X_tab], y, epochs=5, batch_size=16)

save_path = "C:/Users/sauha/.gemini/antigravity-ide/scratch/skin_disease_project/multimodal_model.h5"
model.save(save_path)
print(f"Model trained and saved successfully to {save_path}")
