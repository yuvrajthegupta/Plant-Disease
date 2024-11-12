import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load the model   
import gdown
url = 'https://drive.google.com/uc?export=download&id=15cBqhlxAWxfID4oldyynC9tnO60l_8md'

# Download the model file

def download_model():
    # Check if the model file already exists
    if not os.path.exists('best_model_accuracy.keras'):
        print("Model not found locally. Downloading...")
        gdown.download(url, 'best_model_accuracy.keras', quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")

# Download the model only if it doesn't exist
download_model()
# Direct Google Drive download URL

new_model = tf.keras.models.load_model('best_model_accuracy.keras')
st.title('Plant Disease Detection Model')
st.write("Upload an image to predict its class")
uploaded_image = st.file_uploader("Choose an image...",type=["jpg", "jpeg", "png"])
class_names = ["Apple healthy",
    "Apple scab",
    "Black rot",
    "Cedar apple rust"]

def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image and resize it
    image = load_img(image_path, target_size=target_size)
    
    # Convert the image to a NumPy array
    image_array = img_to_array(image)
    
    # Rescale pixel values to [0, 1] by dividing by 255
    image_array /= 255.0

    return image_array

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)
            # Make a prediction using the pre-trained model
            result = new_model.predict(img_array[np.newaxis, ...])
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')

