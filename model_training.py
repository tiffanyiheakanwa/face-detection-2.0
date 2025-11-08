"""
Facial Emotion Recognition Model Training
This script trains a CNN model to detect 7 emotions from facial images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 48  # FER2013 images are 48x48 pixels
BATCH_SIZE = 64
EPOCHS = 50
TRAIN_DIR = 'train'
TEST_DIR = 'test'

# Emotion labels (7 categories)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print("="*60)
print("FACIAL EMOTION RECOGNITION MODEL TRAINING")
print("="*60)
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Emotions: {EMOTIONS}")
print("="*60)

# Step 1: Data Preparation
print("\n[STEP 1] Preparing data...")

# Data augmentation for training (creates variations of images to improve learning)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to 0-1
    rotation_range=10,            # Randomly rotate images
    width_shift_range=0.1,       # Randomly shift horizontally
    height_shift_range=0.1,      # Randomly shift vertically
    horizontal_flip=True,        # Randomly flip images
    zoom_range=0.1,              # Randomly zoom
    fill_mode='nearest'
)

# For test data, only rescale (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',      # FER2013 is grayscale
    class_mode='categorical',    # Multi-class classification
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")
print(f"✓ Classes found: {train_generator.class_indices}")

# Step 2: Build the CNN Model
print("\n[STEP 2] Building CNN model...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # First convolutional block
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Fourth convolutional block
    layers.Conv2D(512, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    # Output layer (7 emotions)
    layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built successfully!")
model.summary()

# Step 3: Training Callbacks
print("\n[STEP 3] Setting up training callbacks...")

# Early stopping: stop training if no improvement
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when learning plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Step 4: Train the Model
print("\n[STEP 4] Training the model...")
print("This may take 30-60 minutes depending on your hardware...")
print("-"*60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# Step 5: Evaluate the Model
print("\n[STEP 5] Evaluating model...")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {test_accuracy*100:.2f}%")
print(f"FINAL TEST LOSS: {test_loss:.4f}")
print(f"{'='*60}")

# Step 6: Save the Model
print("\n[STEP 6] Saving model...")

model.save('face_emotionModel.h5')
print("✓ Model saved as 'face_emotionModel.h5'")

# Step 7: Plot Training History (optional - saves as image)
print("\n[STEP 7] Generating training graphs...")

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✓ Training history saved as 'training_history.png'")

print("\n" + "="*60)
print("TRAINING COMPLETE! ✓")
print("="*60)
print(f"Model file: face_emotionModel.h5")
print(f"Final accuracy: {test_accuracy*100:.2f}%")
print("\nYou can now proceed to building the Flask web app!")