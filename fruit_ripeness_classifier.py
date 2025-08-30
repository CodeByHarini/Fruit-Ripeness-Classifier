import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# ------------------------------
# STEP 1 — SETUP & CHECK DATASET
# ------------------------------
base_dir = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Dataset"
train_dir = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Dataset\Train"
test_dir = r"C:\Users\hello\Downloads\Fruit Ripeness Classifier\Dataset\Test"

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    print("❌ ERROR: Dataset folder not found!")
    sys.exit(1)

print("✅ Dataset found!")
print(f"Training folder: {train_dir}")
print(f"Testing folder : {test_dir}")

# ------------------------------
# STEP 2 — IMAGE PREPROCESSING
# ------------------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# Rescaling only for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training"
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation"
)

# Testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ------------------------------
# STEP 3 — BUILD IMPROVED CNN MODEL
# ------------------------------
model = Sequential([
    # First Convolution Block
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Second Convolution Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    # Third Convolution Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Flatten + Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile model with lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("\n✅ Model Summary:")
model.summary()

# ------------------------------
# STEP 4 — TRAIN THE MODEL
# ------------------------------
# Callbacks for better performance
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(base_dir, "best_fruit_model.h5"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=30,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ------------------------------
# STEP 5 — SAVE FINAL MODEL
# ------------------------------
MODEL_PATH = os.path.join(base_dir, "fruit_ripeness_model_final.h5")
model.save(MODEL_PATH)
print(f"\n✅ Final Model saved at: {MODEL_PATH}")

# ------------------------------
# STEP 6 — PLOT TRAINING RESULTS
# ------------------------------
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
