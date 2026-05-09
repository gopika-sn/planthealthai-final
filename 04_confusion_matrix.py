import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("models/mobilenet_best.h5")

# Dataset path
dataset_path = "data/color"

# Class names
class_names = sorted(os.listdir(dataset_path))

# Test data
datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predictions
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
fig, ax = plt.subplots(figsize=(18,18))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=90,
    ax=ax,
    colorbar=False
)

plt.tight_layout()

# Save image
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/confusion_matrix.png")

print("Confusion matrix saved successfully!")
