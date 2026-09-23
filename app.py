import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "brain_tumor_rnn.keras"
IMG_SIZE = 64


# ============================================================
# CLASS NAMES
# ============================================================
# IMPORTANT:
# Put your actual class names here in the SAME ORDER
# used during model training.

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

    st.error("Unable to load the model.")

    st.write(
        "Make sure the model file is present in your "
        "GitHub repository:"
    )

    st.code(MODEL_PATH)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("Brain Tumor AI Classifier")

st.caption(
    "RNN-based deep learning application for brain image classification"
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "RNN"
    )

with col2:
    st.metric(
        "Image Size",
        "64 x 64"
    )

with col3:
    st.metric(
        "RNN Input",
        "64 x 192"
    )

with col4:
    st.metric(
        "Task",
        "Classification"
    )


st.divider()


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.subheader("Upload Brain Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# WHEN IMAGE IS UPLOADED
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # OPEN IMAGE
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
            "Original image: "
            + str(original_image.width)
            + " x "
            + str(original_image.height)
        )


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    resized_image = original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )


    img_array = np.array(
        resized_image
    ).astype("float32")


    # Normalize pixel values
    img_array = img_array / 255.0


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------------
    # RESHAPE FOR RNN
    # --------------------------------------------------------
    #
    # Original:
    # 64 x 64 x 3
    #
    # RNN input:
    # 1 x 64 x 192
    #
    # 64 x (64 x 3) = 64 x 192
    # --------------------------------------------------------

    img_array = img_array.reshape(
        1,
        64,
        192
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing image..."):

        prediction = model.predict(
            img_array,
            verbose=0
        )


    # --------------------------------------------------------
    # GET PREDICTION
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(prediction[0])
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = float(
        prediction[0][predicted_index]
    )


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    with result_col:

        st.subheader("AI Prediction")

        st.success(
            "Predicted Class: "
            + predicted_class
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            confidence
        )


        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        st.write("### Class Probabilities")


        for i in range(len(class_names)):

            probability = float(
                prediction[0][i]
            )


            st.write(
                f"**{class_names[i]}**: "
                f"{probability * 100:.2f}%"
            )


            st.progress(
                probability
            )


    # ========================================================
    # IMAGE PROCESSING INFORMATION
    # ========================================================

    st.divider()

    st.subheader("Image Processing")

    process1, process2, process3 = st.columns(3)


    with process1:

        st.info(
            "Original Image\n\n"
            + str(original_image.width)
            + " x "
            + str(original_image.height)
        )


    with process2:

        st.info(
            "Resized Image\n\n"
            "64 x 64"
        )


    with process3:

        st.info(
            "RNN Input Shape\n\n"
            "1 x 64 x 192"
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("About the Project")

st.write(
    "This project uses a Recurrent Neural Network (RNN) "
    "for brain image classification."
)

st.write(
    "The uploaded image is resized to 64 x 64 pixels, "
    "normalized between 0 and 1, and reshaped into "
    "the input format required by the trained RNN model."
)


# ============================================================
# WORKFLOW
# ============================================================

st.subheader("Prediction Workflow")

step1, step2, step3, step4 = st.columns(4)


with step1:

    st.write("**1. Upload**")

    st.caption(
        "Select a brain image."
    )


with step2:

    st.write("**2. Preprocess**")

    st.caption(
        "Resize and normalize the image."
    )


with step3:

    st.write("**3. RNN Analysis**")

    st.caption(
        "The trained model analyzes the image."
    )


with step4:

    st.write("**4. Prediction**")

    st.caption(
        "Class and confidence are displayed."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application is for educational and research purposes "
    "and should not be considered a medical diagnosis."
)


st.caption(
    "Brain Tumor AI Classifier | RNN Deep Learning Project"
)
