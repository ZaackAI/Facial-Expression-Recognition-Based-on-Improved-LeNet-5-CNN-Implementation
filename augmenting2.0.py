import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/CK+_resized"  # Your raw CK+ dataset
OUTPUT_DIR = "data/CK+_augmented"  # Will contain 1000 images per class
TARGET_COUNT = 1000  # As specified in paper's Table 4

# Paper-specified augmentations
augmenter = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    fill_mode='nearest'
)

# Create augmented dataset
for class_dir in tqdm(os.listdir(INPUT_DIR)):
    class_path = os.path.join(INPUT_DIR, class_dir)
    output_class_path = os.path.join(OUTPUT_DIR, class_dir)
    os.makedirs(output_class_path, exist_ok=True)
    
    # Count existing images
    existing_images = [f for f in os.listdir(class_path) if f.endswith(('.png','.jpg'))]
    needed = max(0, TARGET_COUNT - len(existing_images))
    
    # Copy originals first
    for img_file in existing_images:
        src = os.path.join(class_path, img_file)
        dst = os.path.join(output_class_path, f"orig_{img_file}")
        cv2.imwrite(dst, cv2.imread(src, cv2.IMREAD_GRAYSCALE))
    
    # Generate augmented samples
    pbar = tqdm(total=needed, desc=f"Augmenting {class_dir}")
    while pbar.n < needed:
        for img_file in existing_images:
            img = cv2.imread(os.path.join(class_path, img_file), cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            
            # Generate augmented version
            aug_img = augmenter.random_transform(img)
            cv2.imwrite(os.path.join(output_class_path, f"aug_{pbar.n}.png"), aug_img)
            
            pbar.update(1)
            if pbar.n >= needed:
                break
    pbar.close()