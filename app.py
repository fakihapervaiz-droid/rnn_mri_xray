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
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "brain_tumor_rnn.keras"

IMG_SIZE = 64


# ============================================================
# CLASS NAMES
# ============================================================
# IMPORTANT:
# Replace these with the EXACT class names and order
# that you used while training your RNN model.

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
        "The brain tumor model could not be loaded."
    )

    st.info(
        "Make sure the file 'brain_tumor_rnn.keras' "
        "is uploaded to the same GitHub repository as app.py."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("Brain Tumor AI Classifier")

st.write(
    "RNN-based deep learning application for "
    "brain image classification."
)


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("Model Information")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="Model",
        value="RNN"
    )


with col2:

    st.metric(
        label="Image Size",
        value="64 x 64"
    )


with col3:

    st.metric(
        label="Input Shape",
        value="64 x 192"
    )


with col4:

    st.metric(
        label="Task",
        value="Classification"
    )


st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Brain Image")

uploaded_file = st.file_uploader(
    "Choose a brain image",
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

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    image_col, result_col = st.columns(
        [1, 1]
    )


    with image_col:

        st.subheader("Input Image")

        st.image(
            original_image,
            use_container_width=True
        )

        st.caption(
            "Original image size: "
            + str(original_image.width)
            + " x "
            + str(original_image.height)
        )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    img = original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )


    img_array = np.array(
        img
    ).astype("float32")


    # Normalize
    img_array = img_array / 255.0


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------------
    # RNN INPUT SHAPE
    # --------------------------------------------------------
    #
    # Image:
    # 64 x 64 x 3
    #
    # Reshape:
    # 1 x 64 x 192
    #
    # This matches your original prediction code.
    # --------------------------------------------------------

    img_array = img_array.reshape(
        1,
        64,
        64 * 3
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing image..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )


    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col:

        st.subheader("AI Prediction")


        st.success(
            "Predicted Class: "
            + predicted_class
        )


        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )


        st.progress(
            float(
                prediction[0][predicted_index]
            )
        )


        # ----------------------------------------------------
        # ALL CLASS PROBABILITIES
        # ----------------------------------------------------

        st.write("### Class Probabilities")


        for i, class_name in enumerate(
            class_names
        ):

            probability = (
                prediction[0][i]
                * 100
            )


            st.write(
                f"**{class_name}**: "
                f"{probability:.2f}%"
            )


            st.progress(
                float(
                    prediction[0]

