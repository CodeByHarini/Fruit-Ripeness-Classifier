import tensorflow as tf

# Load the model from Models folder
model = tf.keras.models.load_model(
    r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Models\fruit_ripeness_model_final.h5"
)

# Example: check model summary
model.summary()
