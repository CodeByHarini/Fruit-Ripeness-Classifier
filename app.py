import os
import numpy as np
import tensorflow as tf
from PIL import Image
from pyngrok import ngrok, exception
import streamlit as st

# ------------------------------
# Paths
# ------------------------------
h5_model_path = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Backup\Models\fruit_ripeness_model_final.h5"
tflite_model_path = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Backup\Models\fruit_ripeness_model_final.tflite"

# ------------------------------
# Step 1: Convert .h5 to .tflite if missing
# ------------------------------
if not os.path.exists(tflite_model_path):
    if not os.path.exists(h5_model_path):
        raise FileNotFoundError(f"H5 model not found at {h5_model_path}")
    st.info("TFLite model not found. Converting from H5...")
    model = tf.keras.models.load_model(h5_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    st.success("TFLite model created successfully!")
else:
    st.info("TFLite model exists. Loading...")

# ------------------------------
# Step 2: Load TFLite model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# Step 3: Start Ngrok safely
# ------------------------------
port = 8501
public_url = None
try:
    ngrok.kill()  # kill existing tunnels
    public_url = ngrok.connect(port)
    print(f"🌐 Live App URL: {public_url}")
except exception.PyngrokNgrokError as e:
    st.warning(f"Ngrok error: {e}")

# ------------------------------
# Step 4: Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fruit Ripeness Classifier")
st.title("🍎 Fruit Ripeness Classifier")

if public_url:
    st.markdown(f"🌐 **Live App URL:** [{public_url}]({public_url})")

st.write("Upload an image of a fruit to predict its ripeness.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image to match model input
    img_size = input_details[0]['shape'][1]  # assumes square input
    image_resized = image.resize((img_size, img_size))
    input_data = np.array(image_resized, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)  # add batch dimension

    # Set tensor and invoke interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process prediction
    predicted_class = np.argmax(output_data[0])
    confidence = np.max(output_data[0]) * 100

    # Map class index to labels (adjust according to your training)
    class_labels = ["Unripe", "Ripe", "Overripe"]
    st.success(f"Prediction: **{class_labels[predicted_class]}** ({confidence:.2f}%)")
