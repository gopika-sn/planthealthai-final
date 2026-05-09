import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌿",
    layout="centered"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #081c15,
        #1b4332,
        #2d6a4f
    );
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #d8f3dc;
    margin-top: 20px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #b7e4c7;
    margin-bottom: 25px;
}

.result-card {
    background: rgba(255,255,255,0.08);
    border: 2px solid #95d5b2;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    color: white;
    font-size: 28px;
    font-weight: bold;
}

.confidence-card {
    background: rgba(0,0,0,0.25);
    border-radius: 20px;
    padding: 15px;
    text-align: center;
    color: #d8f3dc;
    font-size: 22px;
    font-weight: bold;
}

.label-text {
    color: #95d5b2;
    font-size: 16px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- CLASS LABELS ----------------
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenet_final.h5",
        compile=False
    )

# ---------------- FORMAT LABEL ----------------
def clean_label(label):
    label = label.replace(
        "___",
        " - "
    )

    label = label.replace(
        "_",
        " "
    )

    return label

# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-title'>🌿 PlantHealth AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Smart Plant Disease Detection System</div>",
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📷 Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICT ----------------
if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Smaller image for screenshot fitting
    st.image(
        image,
        caption="Uploaded Plant Image",
        width=350
    )

    # Model preprocessing
    image = image.resize(
        (224, 224)
    )

    img_array = np.array(
        image
    ).astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction) * 100
    )

    predicted_class = clean_label(
        CLASS_NAMES[predicted_index]
    )

    # Prediction card
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='result-card'>
            <div class='label-text'>
                Prediction
            </div>
            {predicted_class}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence card
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='confidence-card'>
            Confidence: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # Alerts
    if confidence < 70:
        st.warning(
            "Try uploading a closer leaf-only image for better accuracy."
        )

    elif confidence > 90:
        st.success(
            "High confidence prediction."
        )
