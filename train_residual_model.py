import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ================= SETTINGS =================
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50

raw_path = "Dataset/raw-890"
ref_path = "Dataset/reference-890"

# ================= LOAD DATA =================
def load_images(raw_dir, ref_dir):
    raw_images = sorted(glob(os.path.join(raw_dir, "*.png")))
    ref_images = sorted(glob(os.path.join(ref_dir, "*.png")))

    X = []
    Y = []

    for r, t in zip(raw_images, ref_images):
        raw = cv2.imread(r)
        ref = cv2.imread(t)

        raw = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
        ref = cv2.resize(ref, (IMG_SIZE, IMG_SIZE))

        raw = raw / 255.0
        ref = ref / 255.0

        X.append(raw)
        Y.append(ref)

    return np.array(X), np.array(Y)

print("Loading dataset...")
X, Y = load_images(raw_path, ref_path)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

print("Dataset loaded:", X.shape)

# ================= RESIDUAL MODEL =================
def buildResidualModel():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(3, 3, padding='same')(x)

    outputs = Add()([inputs, x])
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs, outputs)
    return model

model = buildResidualModel()
model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])

model.summary()

# ================= TRAIN =================
print("Training started...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# ================= SAVE MODEL =================
model.save("trained_residual_model.h5")

print("Training completed and model saved.")
