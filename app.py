import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pyngrok import ngrok
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
# Kill any existing tunnels first
ngrok.kill()

# Start a new ngrok tunnel for Streamlit
public_url = ngrok.connect(8501)
st.markdown(f"🌐 **Live App URL:** [Click here to open]({public_url})")

# Path to your TFLite model
tflite_model_path = os.path.join(
    os.getcwd(),
    "Models",
    "fruit_ripeness_model_final.tflite"
)

# -------------------------------
# Load TFLite model
# -------------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Helper function: preprocess image
# -------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    input_shape = input_details[0]['shape']  # [1, height, width, 3]
    height, width = input_shape[1], input_shape[2]
    image = image.resize((width, height))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Helper function: make prediction
# -------------------------------
def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]
    confidence = np.max(output_data)
    return predicted_class, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍎 Fruit Ripeness Classifier")
st.write("Upload an image of a fruit to check its ripeness.")

uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Ripeness"):
        predicted_class, confidence = predict(image)
        class_labels = ["Unripe", "Ripe", "Overripe"]
        st.success(f"**Prediction:** {class_labels[predicted_class]}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
