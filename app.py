```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Classification AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .header {
        background: linear-gradient(
            135deg,
            #172554 0%,
            #1e3a8a 50%,
            #2563eb 100%
        );
        padding: 35px 40px;
        border-radius: 18px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.18);
    }

    .header h1 {
        font-size: 34px;
        margin-bottom: 8px;
        font-weight: 700;
    }

    .header p {
        font-size: 16px;
        margin: 0;
        opacity: 0.88;
    }

    /* Section title */
    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #172554;
        margin-bottom: 15px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #eff6ff,
            #dbeafe
        );
        border: 1px solid #bfdbfe;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
    }

    .prediction-label {
        color: #475569;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .prediction-name {
        color: #1d4ed8;
        font-size: 30px;
        font-weight: 750;
        margin: 8px 0;
    }

    .confidence {
        color: #334155;
        font-size: 18px;
        font-weight: 600;
    }

    /* Information cards */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
    }

    .info-title {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 5px;
    }

    .info-value {
        font-size: 20px;
        font-weight: 700;
        color: #172554;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "brain_tumor_rnn.keras"

IMG_SIZE = 64

# IMPORTANT:
# Replace these with the exact class names from your dataset.
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
    model_loaded = True

except Exception as e:
    model_loaded = False
    model = None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

    <h1>Brain Tumor Classification AI</h1>

    <p>
        RNN-based medical image classification system for
        automated brain tumor pattern recognition.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL STATUS
# ============================================================

if not model_loaded:

    st.error(
        "Model could not be loaded. "
        "Make sure 'brain_tumor_rnn.keras' is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">MODEL</div>
        <div class="info-value">RNN</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">IMAGE SIZE</div>
        <div class="info-value">64 × 64</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-title">INPUT FORMAT</div>
        <div class="info-value">MRI / Brain Image</div>
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
    "Choose a brain image for classification",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a brain MRI/image supported by your trained dataset."
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image_col, result_col = st.columns([1.05, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_col:

        st.markdown(
            '<div class="section-title">Input Image</div>',
            unsafe_allow_html=True
        )

        original_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            original_image,
            use_container_width=True
        )

        st.caption(
            f"Original image: {original_image.width} × "
            f"{original_image.height} pixels"
        )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    img = original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img_array = np.array(img).astype("float32")

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ========================================================
    # IMPORTANT
    # SAME RESHAPE AS YOUR ORIGINAL CODE
    # ========================================================

    img_array = img_array.reshape(
        1,
        64,
        64 * 3
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

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
        prediction[0][predicted_index] * 100
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col:

        st.markdown(
            '<div class="section-title">Classification Result</div>',
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


        # ----------------------------------------------------
        # CONFIDENCE BAR
        # ----------------------------------------------------

        st.markdown(
            "**Prediction Confidence**"
        )

        st.progress(
            float(prediction[0][predicted_index])
        )

        st.write("")


        # ----------------------------------------------------
        # ALL CLASS PROBABILITIES
        # ----------------------------------------------------

        st.markdown(
            "**Class Probabilities**"
        )

        for i, class_name in enumerate(class_names):

            probability = (
                prediction[0][i] * 100
            )

            st.write(
                f"**{class_name}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                float(prediction[0][i])
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.write("")
st.write("")

st.markdown(
    '<div class="section-title">About This System</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

<b>Brain Tumor Classification AI</b>

<p>
This application uses a trained Recurrent Neural Network (RNN)
to classify brain images according to the classes learned during
model training.
</p>

<p>
The uploaded image is resized to <b>64 × 64</b>, normalized,
converted into a NumPy array, and reshaped into the same input
format used during model training.
</p>

<p style="color:#64748b;">
This application is intended for educational and research purposes
and should not be used as a substitute for professional medical
diagnosis.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Brain Tumor Classification • RNN Deep Learning Project
</div>
""", unsafe_allow_html=True)
```

### Important: change these two things

**1. Your model filename**

If your model is actually named something like:

```text
brain_tumor_rnn_model.keras
```

change:

```python
MODEL_PATH = "brain_tumor_rnn.keras"
```

to your actual filename.

**2. Your actual brain-tumor classes**

Your current code has:

```python
class_names = [
    "Class 0",
    "Class 1",
    "Class 2",
    "Class 3"
]
```

Replace these with the **exact class names from your brain tumor dataset**, in the exact same order used during training.

For example, if your dataset has:

```python
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]
```

then use that.

### One important point about your original code

Your preprocessing is:

```python
64 × 64 × 3
```

and then:

```python
img_array.reshape(1, 64, 64 * 3)
```

so the model receives:

```text
(1, 64, 192)
```

I've kept **exactly that preprocessing** in the UI. This is important because changing the shape could make the deployed model behave differently from your notebook.

Also, because this is a medical-image classifier, the UI labels the result as a **classification prediction**, not a medical diagnosis.
