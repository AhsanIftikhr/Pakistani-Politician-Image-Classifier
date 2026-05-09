import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Pakistani Politician Classifier",
    page_icon="🇵🇰",
    layout="centered"
)

# Load model
model = load_model("resnet50_politician_classifier.h5")

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Title
st.title("🇵🇰 Pakistani Politician Classifier")

st.markdown(
    """
    Upload an image of a Pakistani politician and the AI model
    will predict the politician name.
    """
)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This project uses:
    
    - ResNet50 Transfer Learning
    - TensorFlow & Keras
    - Streamlit Frontend
    
    Developed for ANN Project
    """
)

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Loading spinner
    with st.spinner("Predicting..."):

        # Resize image
        img = image.resize((224, 224))

        # Convert to array
        img_array = np.array(img)

        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess image
        img_array = preprocess_input(img_array)

        # Prediction
        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

    # Prediction Result
    st.success(
        f"✅ Predicted Politician: {class_names[predicted_class]}"
    )

    # Confidence
    st.info(
        f"📊 Confidence: {confidence:.2f}%"
    )

    # Progress bar
    st.progress(float(confidence / 100))

# Footer
st.markdown("---")
st.markdown(
    "Made by Ahsan Wahla with ❤️ using Streamlit & TensorFlow"
)