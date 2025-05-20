import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Enhanced Data Augmentation with Occlusions
def apply_random_occlusion(img):
    """Apply random occlusion during training"""
    img = img.copy()
    occlusions = ["eye", "mouth", "left", None]
    occ_type = np.random.choice(occlusions, p=[0.25, 0.25, 0.25, 0.25])
    
    if occ_type == "eye":
        img[8:18, 6:26] = 0
    elif occ_type == "mouth":
        img[18:28, 8:24] = 0
    elif occ_type == "left":
        img[:, :16] = 0
    return img

# 2. Paper's Exact Model Architecture
def build_model():
    inputs = Input(shape=(32, 32, 1))
    
    # Block 1
    x1 = Conv2D(6, (5,5), activation='relu')(inputs)
    p1 = MaxPooling2D((2,2), strides=2)(x1)
    
    # Block 2 
    x2 = Conv2D(16, (3,3), activation='relu')(p1)
    p2 = MaxPooling2D((2,2), strides=2)(x2)
    
    # Block 3
    x3 = Conv2D(32, (3,3), activation='relu')(p2)
    p3 = MaxPooling2D((2,2), strides=2)(x3)
    
    # Cross-connections
    flat1 = Flatten()(p1)
    flat2 = Flatten()(p2) 
    flat3 = Flatten()(p3)
    merged = concatenate([flat1, flat2, flat3])
    
    # Classifier
    fc1 = Dense(1280, activation='relu')(merged)
    fc2 = Dense(128, activation='relu')(fc1)
    outputs = Dense(7, activation='softmax')(fc2)
    
    return Model(inputs, outputs)

# 3. Training Setup
model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=apply_random_occlusion,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(32,32),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(32,32),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 4. Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

model.save('trained_model_2.h5')

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train') 
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()