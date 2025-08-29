import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load TFLite model
# -------------------------------
tflite_model_path = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Models\fruit_ripeness_model_final.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Helper function: preprocess image
# -------------------------------
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[-1] == 4:  # convert RGBA to RGB
        image = image[..., :3]
    image = image / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image.astype(np.float32)

# -------------------------------
# Helper function: predict
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

        # Map class index to labels (update according to your model's classes)
        class_labels = ["Unripe", "Ripe", "Overripe"]
        st.write(f"**Prediction:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
