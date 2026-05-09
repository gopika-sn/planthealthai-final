import tensorflow as tf

model = tf.keras.models.load_model(
    "mobilenet_final.h5",
    compile=False
)

model.save("mobilenet_streamlit.keras")

print("DONE")
