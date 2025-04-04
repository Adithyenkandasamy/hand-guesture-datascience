import os
import json
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ======== Config ========
csv_path = "/home/jinwoo/Desktop/hand-guesture-datascience/train.csv"
train_dir = "/home/jinwoo/Desktop/hand-guesture-datascience/train/"
img_size = (128, 128)
batch_size = 32
epochs = 10

# ======== Load and Map Labels ========
df = pd.read_csv(csv_path, sep=";")
image_to_label = {row["Image Name"]: row["label"] for _, row in df.iterrows()}

# Save mapping to JSON
with open("image_labels.json", "w") as json_file:
    json.dump(image_to_label, json_file, indent=4)
print("✅ Saved image-label mapping to image_labels.json")

# Split into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
num_classes = df["label"].nunique()

# ======== Custom Generator ========
class ImageDataGeneratorCSV(Sequence):
    def __init__(self, dataframe, batch_size, img_size, train_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_dir = train_dir
        self.label_map = {label: idx for idx, label in enumerate(df["label"].unique())}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) * 10 / self.batch_size))  # estimate batches

    def __getitem__(self, index):
        images, labels = [], []
        while len(images) < self.batch_size:
            row = self.dataframe.sample(n=1).iloc[0]
            folder_name = row["Image Name"]
            label_name = row["label"]
            folder_path = os.path.join(self.train_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not img_files:
                continue

            img_file = random.choice(img_files)
            img_path = os.path.join(folder_path, img_file)

            try:
                img = load_img(img_path, target_size=self.img_size)
                img = img_to_array(img) / 255.0
                images.append(img)
                labels.append(self.label_map[label_name])
            except Exception as e:
                print(f"⚠️ Failed to load image {img_path}: {e}")

        return np.array(images), tf.keras.utils.to_categorical(labels, num_classes)

    def on_epoch_end(self):
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

# ======== Data Generators ========
train_generator = ImageDataGeneratorCSV(train_df, batch_size, img_size, train_dir)
val_generator = ImageDataGeneratorCSV(val_df, batch_size, img_size, train_dir)

# ======== CNN Model ========
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ======== Train Model ========
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# ======== Save Model ========
model.save("gesture_model.h5")
print("✅ Model saved as gesture_model.h5")
