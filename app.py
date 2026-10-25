import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="AI Skin Disease Detection",
    page_icon="📷",
    layout="centered"
)

# Website heading
st.title("AI-Based Skin Disease Detection")
st.write(
    "An educational image-classification project "
    "using a trained AI model."
)

st.warning(
    "Educational use only. This model is not a medical "
    "diagnostic tool. Do not use its predictions to make "
    "healthcare decisions."
)

# Load the trained model
@st.cache_resource
def load_skin_model():
    return tf.keras.models.load_model(
        "skin_disease_model.keras"
    )

try:
    model = load_skin_model()
    st.success("AI model loaded successfully!")
except Exception as e:
    st.error("Could not load the AI model.")
    st.code(str(e))
    st.stop()

# Choose camera or image upload
input_method = st.radio(
    "Choose how to provide an image:",
    ["Take a photo", "Upload a photo"]
)

photo = None

if input_method == "Take a photo":
    photo = st.camera_input("Take a skin photo")
else:
    photo = st.file_uploader(
        "Upload a skin image",
        type=["jpg", "jpeg", "png"]
    )

# Process the image
if photo is not None:
    try:
        image = Image.open(photo).convert("RGB")

        st.subheader("Your image")
        st.image(
            image,
            caption="Image ready for analysis",
            width=350
        )

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Resize to match model input
                resized_image = image.resize((224, 224))

                # Convert image to a NumPy array
                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                # Apply MobileNetV2 preprocessing
                image_array = (
                    tf.keras.applications.mobilenet_v2
                    .preprocess_input(image_array)
                )

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Get prediction
                prediction = float(
                    model.predict(
                        image_array,
                        verbose=0
                    )[0][0]
                )

            # Model labels: Acne = 0, Eczema = 1
            if prediction >= 0.5:
                result = "Eczema"
            else:
                result = "Acne"

            st.subheader("Experimental AI result")
            st.write(f"**Predicted class: {result}**")

            st.caption(
                "This is an experimental classification, "
                "not a diagnosis. The result may be incorrect."
            )

    except Exception as e:
        st.error("Unable to analyze this image.")
        st.code(str(e))
