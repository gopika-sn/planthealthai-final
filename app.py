import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌿",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c7744);
}

.main-title {
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
}

.sub-title {
    text-align:center;
    color:#d9fdd3;
    font-size:18px;
    margin-bottom:20px;
}

.result-box {
    background: rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 20px;
    text-align:center;
    color:white;
    font-size:28px;
    font-weight:bold;
    border:1px solid rgba(255,255,255,0.2);
}

.conf-box {
    background: rgba(0,0,0,0.25);
    border-radius: 12px;
    padding: 12px;
    text-align:center;
    color:#d9fdd3;
    font-size:22px;
    font-weight:bold;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Classes ----------------
CLASS_NAMES = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Rust",
    "Apple - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Healthy"
]


# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenet_streamlit.h5",
        compile=False
    )

model = load_model()


# ---------------- Prediction Function ----------------
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return CLASS_NAMES[predicted_class], confidence


# ---------------- UI ----------------
st.markdown('<p class="main-title">🌿 PlantHealth AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Plant Disease Detection using Deep Learning</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Plant Image", width=350)

    disease, confidence = predict_image(image)

    st.markdown(
        f'<div class="result-box">{disease}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="conf-box">Confidence: {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )

    if confidence < 70:
        st.warning("Try uploading a closer leaf-only image for better accuracy.")
