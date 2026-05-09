import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load model
model = tf.keras.models.load_model("models/mobilenet_best.h5")

# Dataset path
dataset_path = "data/color"

# Class names
class_names = sorted(os.listdir(dataset_path))

# Data
datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)

# True labels
y_true = test_data.classes

# Report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)