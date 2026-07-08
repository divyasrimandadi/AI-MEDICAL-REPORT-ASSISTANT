import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

####################################################
# RANDOM SEED
####################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

####################################################
# CONFIGURATION
####################################################

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

INITIAL_EPOCHS = 5

FINE_TUNE_EPOCHS = 20

LEARNING_RATE = 1e-4

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(ROOT_DIR)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "chest-xray"
)

TRAIN_DIR = os.path.join(DATASET_PATH,"train")

VAL_DIR = os.path.join(DATASET_PATH,"val")

TEST_DIR = os.path.join(DATASET_PATH,"test")

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "model"
)

os.makedirs(MODEL_DIR,exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "medical_model.keras"
)

print("="*70)
print("AI MEDICAL REPORT ASSISTANT")
print("="*70)

print("TensorFlow :",tf.__version__)

if tf.config.list_physical_devices("GPU"):
    print("GPU Detected")
else:
    print("Running on CPU")

####################################################
# DATA AUGMENTATION
####################################################

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    shear_range=0.2,

    horizontal_flip=True,

    fill_mode="nearest"

)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

####################################################
# DATA LOADERS
####################################################

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True

)

validation_generator = valid_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

test_generator = valid_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=1,

    class_mode="binary",

    shuffle=False

)

print()

print(train_generator.class_indices)

class_names = list(train_generator.class_indices.keys())

with open(
    os.path.join(MODEL_DIR,"class_names.txt"),
    "w"
) as f:

    for cls in class_names:

        f.write(cls+"\n")

####################################################
# CLASS WEIGHTS
####################################################

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_generator.classes),

    y=train_generator.classes

)

class_weights = dict(enumerate(weights))

print()

print("Class Weights")

print(class_weights)

####################################################
# BUILD EFFICIENTNET MODEL
####################################################

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dense(
    512,
    activation="relu"
)(x)

x = Dropout(0.50)(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.30)(x)

outputs = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(inputs, outputs)

####################################################
# COMPILE MODEL
####################################################

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

)

model.summary()

####################################################
# CALLBACKS
####################################################

early_stop = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=5,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=2,

    min_lr=1e-7,

    verbose=1

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1

)

callbacks = [

    early_stop,

    reduce_lr,

    checkpoint

]

####################################################
# INITIAL TRAINING
####################################################

print("\n")
print("="*60)
print("INITIAL TRAINING")
print("="*60)

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks,

    class_weight=class_weights,

    verbose=1

)

####################################################
# FINE TUNING
####################################################

print("\n")
print("="*60)
print("FINE TUNING")
print("="*60)

# Unfreeze the base model
base_model.trainable = True

# Freeze first layers and train only last layers
for layer in base_model.layers[:-80]:
    layer.trainable = False

####################################################
# RECOMPILE
####################################################

model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.AUC(name="auc"),

        tf.keras.metrics.Precision(name="precision"),

        tf.keras.metrics.Recall(name="recall")

    ]

)

####################################################
# TRAIN AGAIN
####################################################

history_fine = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks,

    class_weight=class_weights,

    verbose=1

)

####################################################
# LOAD BEST MODEL
####################################################

print("\nLoading Best Model...\n")

model = tf.keras.models.load_model(MODEL_PATH)

####################################################
# MODEL EVALUATION
####################################################

print("\n")
print("="*60)
print("MODEL EVALUATION")
print("="*60)

loss, accuracy, auc_score, precision, recall = model.evaluate(
    test_generator,
    verbose=1
)

print(f"\nTest Loss      : {loss:.4f}")
print(f"Test Accuracy  : {accuracy:.4f}")
print(f"Test AUC       : {auc_score:.4f}")
print(f"Test Precision : {precision:.4f}")
print(f"Test Recall    : {recall:.4f}")

####################################################
# PREDICTIONS
####################################################

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)

y_prob = predictions.flatten()

y_pred = (y_prob > 0.5).astype(int)

y_true = test_generator.classes

####################################################
# METRICS
####################################################

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

print("\n")
print("="*60)
print("FINAL METRICS")
print("="*60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc_score:.4f}")

####################################################
# CLASSIFICATION REPORT
####################################################

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

with open(
    os.path.join(
        MODEL_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

####################################################
# CONFUSION MATRIX
####################################################

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(7,7))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.jpg"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

####################################################
# ROC CURVE
####################################################

fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "roc_curve.jpg"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

####################################################
# PRECISION RECALL CURVE
####################################################

precision_curve, recall_curve, _ = precision_recall_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7,6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid(True)

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "precision_recall.jpg"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()