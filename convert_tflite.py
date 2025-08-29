import tensorflow as tf
import os

# Correct path to the H5 model in Models folder
h5_path = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Models\fruit_ripeness_model_final.h5"

# Check if the file exists
if not os.path.exists(h5_path):
    raise FileNotFoundError(f"❌ H5 model not found at: {h5_path}")

# Load the model
model = tf.keras.models.load_model(h5_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
tflite_path = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Models\fruit_ripeness_model_final.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved at: {tflite_path}")
