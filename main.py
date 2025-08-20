import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("CNN.keras")

model = load_cnn_model()

# Define class names in the same order as your training generator
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Update if needed

st.title("🌸 Flower Classification App")
st.write("Upload a flower image and get prediction probabilities for each class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(320, 240))  # same as training
    img_array = image.img_to_array(img) / 255.0  # rescale like in ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Predict
    predictions = model.predict(img_array)[0]

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Show prediction probabilities
    st.subheader("Prediction Probabilities")
    for i, prob in enumerate(predictions):
        st.write(f"**{class_names[i]}**: {prob:.4f}")

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions)
    ax.set_ylabel("Probability")
    ax.set_title("Softmax Output")
    st.pyplot(fig)
