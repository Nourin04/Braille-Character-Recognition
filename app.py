import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("best_braille_model_with_class_weights.keras")

# Load Label Encoder (assuming it's saved or defined)
le = LabelEncoder()
le.classes_ = np.array([chr(i) for i in range(65, 91)])  # A-Z ASCII uppercase letters

# Streamlit app setup
st.title("Braille Character Recognition")

# Upload an image file
uploaded_file = st.file_uploader("Choose a Braille image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Preprocess the image
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    
    # Predict the character
    prediction = model.predict(img)
    pred_class = np.argmax(prediction)
    predicted_character = le.inverse_transform([pred_class])[0]
    
    # Show the predicted character
    st.write(f"🔠 Predicted Braille Character: {predicted_character}")
