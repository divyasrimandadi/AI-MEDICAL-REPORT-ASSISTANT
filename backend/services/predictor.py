import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore") 

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------
# Paths
# ---------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "medical_model.keras"
)

CLASS_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "class_names.txt"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

print("Loading AI Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ---------------------------------------------------
# Load Class Names
# ---------------------------------------------------

with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print(class_names)

# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img = np.array(image)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

def predict_image(image: Image.Image):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:

        predicted_class = class_names[1]

        confidence = probability

    else:

        predicted_class = class_names[0]

        confidence = 1 - probability

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2)
    }