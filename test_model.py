import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# 1. Fixed occlusion function (modified to handle batch processing)
def apply_occlusion(imgs, occ_type=None):
    """Apply specific occlusion type to a batch of images"""
    imgs = imgs.copy()
    if occ_type == "eye":
        imgs[:, 8:18, 6:26, :] = 0  # Eye region
    elif occ_type == "mouth":
        imgs[:, 18:28, 8:24, :] = 0  # Mouth region
    elif occ_type == "left":
        imgs[:, :, :16, :] = 0  # Left half
    return imgs

# 2. Load model
#model = tf.keras.models.load_model('paper_replication_model.h5') 
model = tf.keras.models.load_model('best_model.h5') 

# 3. Data generator (without preprocessing_function)
test_datagen = ImageDataGenerator(rescale=1./255)

# 4. Evaluation function (modified)
def evaluate_test_set(occ_type=None):
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(32, 32),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate with optional occlusion
    if occ_type:
        print(f"\nEvaluating with {occ_type} occlusion...")
        total_samples = 0
        correct_predictions = 0
        
        for i in range(len(test_generator)):
            x, y = test_generator[i]
            x_occluded = apply_occlusion(x, occ_type)
            predictions = model.predict(x_occluded, verbose=0)
            correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            total_samples += len(y)
            
            if total_samples >= test_generator.samples:
                break
        
        acc = correct_predictions / total_samples
        print(f"{occ_type} occlusion: {acc:.2%} (n={total_samples})")
    else:
        loss, acc = model.evaluate(test_generator, verbose=0)
        print(f"Clean: {acc:.2%} (n={test_generator.samples})")

# 5. Run tests
print("=== Test Results ===")
evaluate_test_set()          # Clean
evaluate_test_set("eye")     # Eye occlusion
evaluate_test_set("mouth")   # Mouth occlusion
evaluate_test_set("left")    # Left occlusion

# Visualization function (modified)
def show_sample_occlusions():
    sample_datagen = ImageDataGenerator(rescale=1./255)
    sample_generator = sample_datagen.flow_from_directory(
        'data/test',
        target_size=(32, 32),
        color_mode='grayscale',
        batch_size=1,
        class_mode='categorical',
        shuffle=True
    )
    
    plt.figure(figsize=(15, 5))
    occlusion_types = [None, "eye", "mouth", "left"]
    
    for i, occ_type in enumerate(occlusion_types):
        img, _ = next(sample_generator)
        if occ_type:
            img = apply_occlusion(img, occ_type)
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img[0].squeeze(), cmap='gray')
        plt.title(f"{occ_type or 'Clean'}")
        plt.axis('off')
    plt.show()

show_sample_occlusions()
