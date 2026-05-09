import tensorflow as tf

model = tf.keras.models.load_model(
    "mobilenet_final.h5",
    compile=False
)

model.save("mobilenet_streamlit.h5", include_optimizer=False)

print("Done")
