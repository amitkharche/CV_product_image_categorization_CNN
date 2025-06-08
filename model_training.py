"""
Train a CNN to classify e-commerce product images.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMG_SIZE = (64, 64)
DATA_PATH = "data/product_labels.csv"
MODEL_DIR = "model"

def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["product_category"])
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    return df, le

def load_images(df):
    images = []
    labels = df["label_id"].values
    for path in df["image_path"]:
        img = load_img(path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images), to_categorical(labels)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df, le = load_data()
    X, y = load_images(df)
    #Straify=y removed due to small dataset
    # X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((*IMG_SIZE, 3), y.shape[1])
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, callbacks=[early_stop])
    model.save(os.path.join(MODEL_DIR, "product_cnn_model.h5"))
    logging.info("✅ Model training and saving complete.")

if __name__ == "__main__":
    main()
