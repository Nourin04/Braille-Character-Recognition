import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the model and label encoder
model = load_model("best_braille_model_with_class_weights.keras")  # Correct model path
le = LabelEncoder()
le.classes_ = np.array([chr(i) for i in range(65, 91)])  # A-Z ASCII uppercase letters

# Title and description for the app
st.title("Braille Character Recognition")
st.write("Upload an image of a Braille character, and the model will predict the corresponding letter.")

# File uploader for image
uploaded_file = st.file_uploader("Choose a Braille image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Check and display the shape of the uploaded image
    st.write(f"Original Image Shape: {img.shape}")  # Check original shape

    # Preprocess the image before feeding it into the model
    img_resized = cv2.resize(img, (28, 28))  # Resize to match the input shape of the model (28x28)
    st.write(f"Resized Image Shape: {img_resized.shape}")  # Check resized shape
    
    img_resized = img_resized / 255.0  # Normalize the image
    st.write(f"Normalized Image Shape: {img_resized.shape}")  # Check shape after normalization
    
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension to make it (1, 28, 28, 3)
    st.write(f"Final Image Shape: {img_resized.shape}")  # Check final shape
    
    # Predict the character
    try:
        prediction = model.predict(img_resized)
        predicted_class = np.argmax(prediction)
        predicted_character = le.inverse_transform([predicted_class])[0]

        # Display the result
        st.write(f"🔠 Predicted Braille Character: {predicted_character}")

        # Optional: Display probability of prediction
        st.write(f"Prediction probabilities: {prediction[0]}")
        
        # Plot the image again with prediction result
        fig, ax = plt.subplots()
        ax.imshow(img_resized[0])
        ax.set_title(f"Predicted: {predicted_character}")
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error during prediction: {e}")
