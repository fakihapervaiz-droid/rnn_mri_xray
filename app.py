
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    page_icon="MRI",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f6f8fc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.header {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #2563eb
    );
    padding: 38px;
    border-radius: 20px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
}

.header h1 {
    font-size: 36px;
    margin: 0 0 10px 0;
    font-weight: 750;
}

.header p {
    font-size: 16px;
    margin: 0;
    opacity: 0.9;
}

.section-title {
    color: #0f172a;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 17px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
}

.prediction-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 17px;
    padding: 28px;
    text-align: center;
    margin-bottom: 18px;
}

.prediction-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-name {
    color: #1d4ed8;
    font-size: 29px;
    font-weight: 750;
    margin: 8px 0;
}

.confidence {
    color: #334155;
    font-size: 18px;
    font-weight: 600;
}

.stat {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 17px;
    text-align: center;
}

.stat-title {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
}

.stat-value {
    color: #0f172a;
    font-size: 19px;
    font-weight: 700;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "brain_tumor_rnn.keras"

IMG_SIZE = 64


# IMPORTANT:
# Replace these with the EXACT class names and order
# used when training your model.

class_names = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_brain_model():
    return load_model(MODEL_PATH)


try:
    model = load_brain_model()

except Exception as e:

    st.error(
        "The trained model could not be loaded. "
        "Make sure the model file is in the same GitHub repository "
        "as app.py."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

    <h1>Brain Tumor AI Classifier</h1>

    <p>
        RNN-based deep learning application for brain image
        classification using automated image preprocessing.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown("""
    <div class="stat">

        <div class="stat-title">
            MODEL
        </div>

        <div class="stat-value">
            RNN
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="stat">

        <div class="stat-title">
            IMAGE SIZE
        </div>

        <div class="stat-value">
            64 x 64
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="stat">

        <div class="stat-title">
            INPUT
        </div>

        <div class="stat-value">
            MRI Image
        </div>

    </div>
    """, unsafe_allow_html=True)


with col4:

    st.markdown("""
    <div class="stat">

        <div class="stat-title">
            TASK
        </div>

        <div class="stat-value">
            Classification
        </div>

    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Brain Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG, PNG or WEBP image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    image_col, result_col = st.columns(
        [1, 1]
    )


    # ========================================================
    # DISPLAY ORIGINAL IMAGE
    # ========================================================

    with image_col:

        st.markdown(
            '<div class="section-title">Input Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            use_container_width=True
        )

        st.caption(
            "Original size: "
            + str(original_image.width)
            + " x "
            + str(original_image.height)
            + " pixels"
        )


    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    img = original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )


    img_array = np.array(
        img
    ).astype("float32")


    # Normalize image
    img_array = img_array / 255.0


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # ========================================================
    # RNN INPUT SHAPE
    #
    # Original image:
    # 64 x 64 x 3
    #
    # Reshaped for RNN:
    # 1 x 64 x 192
    # ========================================================

    img_array = img_array.reshape(
        1,
        64,
        64 * 3
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with st.spinner("Analyzing image..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )


    predicted_index = np.argmax(
        prediction[0]
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = (
        prediction[0][predicted_index]
        * 100
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    with result_col:

        st.markdown(
            '<div class="section-title">AI Prediction</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    Predicted Class
                </div>

                <div class="prediction-name">
                    {predicted_class}
                </div>

                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write(
            "**Prediction Confidence**"
        )


        st.progress(
            float(
                prediction[0][predicted_index]
            )
        )


        st.write("")


        # ====================================================
        # ALL CLASS PROBABILITIES
        # ====================================================

        st.write(
            "**Class Probabilities**"
        )


        for i, class_name in enumerate(
            class_names
        ):

            probability = (
                prediction[0][i]
                * 100
            )


            st.write(
                f"{class_name}: "
                f"**{probability:.2f}%**"
            )


            st.progress(
                float(
                    prediction[0][i]
                )
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.write("")
st.write("")


st.markdown(
    '<div class="section-title">About the Project</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="card">

<b>Brain Tumor Classification using RNN</b>

<br><br>

This application demonstrates an end-to-end deep learning
workflow for classifying brain images using a Recurrent
Neural Network.

<br><br>

<b>Pipeline:</b>

<br><br>

Image Upload
&nbsp;&rarr;&nbsp;
Image Resize
&nbsp;&rarr;&nbsp;
Normalization
&nbsp;&rarr;&nbsp;
RNN Input
&nbsp;&rarr;&nbsp;
Prediction
&nbsp;&rarr;&nbsp;
Confidence Analysis

<br><br>

<span style="color:#64748b;">
This application is intended for educational and research
purposes. The predictions should not be considered a
medical diagnosis.
</span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Brain Tumor AI Classifier | RNN Deep Learning Project

</div>
""", unsafe_allow_html=True)
