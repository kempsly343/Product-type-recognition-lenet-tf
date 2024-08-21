import json
import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import io
import urllib.request

# Configuration dictionary with image size and class names
CONFIGURATION = {
    "IM_SIZE": 256,
    "CLASS_NAMES": ["ACCESSORIES", "BRACELETS", "CHAIN", "CHARMS", "EARRINGS",
     "ENGAGEMENT RINGS", "ENGAGEMENT SET", "FASHION RINGS", "NECKLACES", "WEDDING BANDS"],
}

# Function to preprocess the image
def preprocess_image(image, im_size):
    image = cv2.resize(image, (im_size, im_size))
    image = image.astype(np.float32) / 255  # Normalize image
    return image

# Function to get prediction from the web service
def get_prediction(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image, CONFIGURATION["IM_SIZE"])

    # Convert image to list and create JSON payload with the 'instances' key
    data = json.dumps({"instances": [preprocessed_image.tolist()]})
    headers = {'Content-Type': 'application/json'}

    # URL of the deployed web service
    scoring_uri = "http://4.153.37.197:80/api/v1/service/product-recognition-ltf/v1/models/lenet_model_tf:predict"

    # If your service requires authentication, use the key
    api_key = "XloekJ9K6oDHc2Hs3aaQVBJw5Dto8gY1"
    headers['Authorization'] = f'Bearer {api_key}'

    # Make the request
    response = requests.post(scoring_uri, data=data, headers=headers)
    prediction = response.json()

    return prediction

# Function to map the prediction to class names
def get_class_name(prediction):
    class_probs = prediction.get('predictions', [])[0]
    max_index = np.argmax(class_probs)
    class_name = CONFIGURATION["CLASS_NAMES"][max_index]
    return class_name

# Streamlit web interface
def main():
    st.title("Product Type Classification")

    st.write("Upload an image or provide an image URL for classification:")

    # Option to upload an image or provide a URL
    choice = st.radio("Choose an option", ('Upload Image', 'Provide Image URL'))

    if choice == 'Upload Image':
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image.convert('RGB'))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            if st.button("Classify Image"):
                prediction = get_prediction(image)
                class_name = get_class_name(prediction)
                st.write(f"Predicted Class: {class_name}")

    elif choice == 'Provide Image URL':
        image_url = st.text_input("Enter the image URL:")
        if st.button("Classify Image"):
            if image_url:
                try:
                    image = load_image_from_url(image_url)
                    st.image(image, caption='Image from URL.', use_column_width=True)
                    
                    prediction = get_prediction(image)
                    class_name = get_class_name(prediction)
                    st.write(f"Predicted Class: {class_name}")
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")

# Function to load image from a URL
def load_image_from_url(image_url):
    resp = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

if __name__ == "__main__":
    main()
