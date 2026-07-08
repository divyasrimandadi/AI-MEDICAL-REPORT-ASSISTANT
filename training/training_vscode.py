#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:




# In[3]:




# In[4]:




# In[5]:




# In[6]:




# In[7]:




# In[8]:




# In[9]:




# In[10]:





# In[11]:




# In[12]:


from google.colab import files
files.upload()


# In[13]:




# In[14]:




# In[15]:




# In[16]:




# In[17]:




# In[18]:


import os

BASE_PATH = "./chest_xray"

train_normal = len(os.listdir(f"{BASE_PATH}/train/NORMAL"))
train_pneumonia = len(os.listdir(f"{BASE_PATH}/train/PNEUMONIA"))

print("NORMAL:", train_normal)
print("PNEUMONIA:", train_pneumonia)


# In[19]:


import matplotlib.pyplot as plt

labels = ['NORMAL', 'PNEUMONIA']
counts = [1341, 3875]

plt.figure(figsize=(6,4))
plt.bar(labels, counts)

plt.title("Label Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")

plt.show()


# In[20]:


import matplotlib.pyplot as plt
import cv2
import os

folder = "./chest_xray/train/NORMAL"

images = os.listdir(folder)[:5]

plt.figure(figsize=(15,5))

for i, img_name in enumerate(images):

    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title("NORMAL")
    plt.axis("off")

plt.tight_layout()
plt.show()


# In[21]:




# In[22]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    "./chest_xray/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "./chest_xray/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    "./chest_xray/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)


# In[24]:


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False


# In[25]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[26]:


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)


# In[28]:


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import tensorflow as tf

IMG_SIZE = 224

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC()
    ]
)

model.summary()


# In[29]:


history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=15
)


# In[30]:


print("Final Training Accuracy:", history.history['accuracy'][-1])


# In[31]:


print("Final Validation Accuracy:", history.history['val_accuracy'][-1])


# In[32]:


print("Final Training AUC:", history.history['auc'][-1])


# In[33]:


print("Final Validation AUC:", history.history['val_auc'][-1])


# In[34]:


print("Best Validation AUC:", max(history.history['val_auc']))


# In[35]:


test_loss, test_accuracy, test_auc = model.evaluate(test_generator)

print("Test Accuracy:", test_accuracy)
print("Test AUC:", test_auc)


# In[36]:


from sklearn.metrics import classification_report

predictions = model.predict(test_generator)

y_pred = (predictions > 0.5).astype(int)

y_true = test_generator.classes

print(classification_report(y_true, y_pred))


# In[37]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[38]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, predictions)

roc_auc = auc(fpr, tpr)

print("ROC AUC Score:", roc_auc)


# In[39]:


model.save("medical_model.keras")


# In[51]:


files.download("medical_model.keras")


# In[41]:


class_names = ['NORMAL', 'PNEUMONIA']

with open("class_names.txt", "w") as f:
    for item in class_names:
        f.write(item + "\n")


# In[42]:


files.download("class_names.txt")


# In[43]:


import json

with open("training_history.json", "w") as f:
    json.dump(history.history, f)


# In[44]:


files.download("training_history.json")


# In[49]:


print(train_generator.class_indices)


# In[48]:


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ----------------------------
# Create Output Folder
# ----------------------------
os.makedirs("training/outputs", exist_ok=True)

# ----------------------------
# Accuracy Graph
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "training/outputs/accuracy_graph.jpg",
    dpi=300,
    format="jpg",
    bbox_inches="tight"
)
plt.close()

# ----------------------------
# Loss Graph
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "training/outputs/loss_graph.jpg",
    dpi=300,
    format="jpg",
    bbox_inches="tight"
)
plt.close()

# ----------------------------
# Label Distribution
# ----------------------------
labels = list(train_generator.class_indices.keys())
counts = [
    len(train_generator.classes[train_generator.classes == i])
    for i in range(len(labels))
]

plt.figure(figsize=(8,5))
plt.bar(labels, counts)
plt.title("Label Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(
    "training/outputs/label_distribution.jpg",
    dpi=300,
    format="jpg",
    bbox_inches="tight"
)
plt.close()

# ----------------------------
# Sample Images
# ----------------------------
plt.figure(figsize=(12,8))

images, labels_batch = next(train_generator)

for i in range(min(6, len(images))):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(labels[np.argmax(labels_batch[i])] if len(labels_batch.shape)>1 else labels[int(labels_batch[i])])
    plt.axis("off")

plt.tight_layout()
plt.savefig(
    "training/outputs/sample_images.jpg",
    dpi=300,
    format="jpg",
    bbox_inches="tight"
)
plt.close()

# ----------------------------
# Predictions
# ----------------------------
test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)

if predictions.shape[1] == 1:

    y_pred = (predictions > 0.5).astype(int).flatten()
    y_scores = predictions.flatten()

else:

    y_pred = np.argmax(predictions, axis=1)
    y_scores = predictions[:,1]

y_true = test_generator.classes

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues")
plt.tight_layout()
plt.savefig(
    "training/outputs/confusion_matrix.jpg",
    dpi=300,
    format="jpg",
    bbox_inches="tight"
)
plt.close()

# ----------------------------
# ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "training/outputs/roc_curve.jpg",
    dpi=300,
    format="jpg",
    bbox_inches="tight"
)
plt.close()

print("="*60)
print("✅ ALL IMAGES SAVED SUCCESSFULLY")
print("="*60)
print("training/outputs/")
print("   accuracy_graph.jpg")
print("   loss_graph.jpg")
print("   label_distribution.jpg")
print("   sample_images.jpg")
print("   confusion_matrix.jpg")
print("   roc_curve.jpg")
print("="*60)


# In[ ]:





# In[ ]:


