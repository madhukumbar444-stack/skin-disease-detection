import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI Skin Detection", page_icon="📷")

st.title("AI-Based Skin Disease Detection")
st.write("Take a photo of a skin area for educational image analysis.")

st.warning(
    "This project is for educational purposes only. "
    "It cannot provide a medical diagnosis."
)

photo = st.camera_input("Take a skin photo")

if photo is not None:
    image = Image.open(photo).convert("RGB")

    st.subheader("Your captured photo")
    st.image(image, caption="Photo captured successfully", width=400)

    st.success("Photo captured! We can connect the AI model next.")
