import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("models/mobilenet_best.h5")

# Dataset
dataset_path = "data/color"

# Classes
class_names = sorted(os.listdir(dataset_path))

# Collect random images
all_images = []

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)

    for file in os.listdir(class_folder)[:5]:
        all_images.append((os.path.join(class_folder, file), class_name))

# Random sample
samples = random.sample(all_images, 10)

# Plot
plt.figure(figsize=(15,12))

for i, (img_path, true_label) in enumerate(samples):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    pred_index = np.argmax(prediction)
    pred_label = class_names[pred_index]

    plt.subplot(2,5,i+1)
    plt.imshow(img)

    plt.title(
        f"True:\n{true_label}\n\nPred:\n{pred_label}",
        fontsize=8
    )

    plt.axis("off")

plt.tight_layout()

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/test_predictions.png")

print("Test predictions saved!")