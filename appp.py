import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------------
# Load Keras .h5 model
# -------------------------------
model_path = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Models\fruit_ripeness_model_final.h5"
model = load_model(model_path)

# -------------------------------
# Helper function: preprocess image
# -------------------------------
def preprocess_image(image, target_size=(150, 150)):
    """
    Convert uploaded PIL image to normalized numpy array ready for prediction.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # normalize
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    return image_array

# -------------------------------
# Helper function: predict
# -------------------------------
def predict(image):
    input_data = preprocess_image(image)
    preds = model.predict(input_data)
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)
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
        # Map class index to labels
        class_labels = ["Unripe", "Ripe", "Overripe"]
        st.write(f"**Prediction:** {class_labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
