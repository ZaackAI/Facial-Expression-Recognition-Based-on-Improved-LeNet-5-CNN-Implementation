import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

def validate_image(img_path):
    """Ensure image is valid and grayscale"""
    try:
        with Image.open(img_path) as img:
            img.verify()
            if img.mode != 'L':
                img = img.convert('L')
            return True
    except Exception as e:
        print(f"Removing corrupt file: {img_path} - {str(e)}")
        try:
            os.remove(img_path)
        except:
            pass
        return False

def create_splits():
    # Configuration
    SOURCE_DIR = "data/CK+_augmented"  # Using augmented dataset
    TEST_DIR = "data/test"
    VAL_DIR = "data/val"
    TRAIN_DIR = "data/train"
    TEST_RATIO = 0.2
    VAL_RATIO = 0.25  # 25% of remaining 80% = 20% of total
    SEED = 42

    # Clean directories
    for dir_path in [TEST_DIR, VAL_DIR, TRAIN_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # Process each class
    for expression in os.listdir(SOURCE_DIR):
        expr_dir = os.path.join(SOURCE_DIR, expression)
        if not os.path.isdir(expr_dir):
            continue

        # Create class directories
        for dir_path in [TEST_DIR, VAL_DIR, TRAIN_DIR]:
            os.makedirs(os.path.join(dir_path, expression), exist_ok=True)

        # Get and validate images
        images = []
        for f in os.listdir(expr_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(expr_dir, f)
                if validate_image(img_path):
                    images.append(f)
        
        # Skip if no valid images
        if len(images) == 0:
            print(f"⚠️ No valid images found for class: {expression}")
            continue

        # Create splits
        try:
            trainval, test = train_test_split(
                images,
                test_size=TEST_RATIO,
                random_state=SEED
            )
            train, val = train_test_split(
                trainval,
                test_size=VAL_RATIO,
                random_state=SEED
            )
        except ValueError as e:
            print(f"⚠️ Error splitting {expression}: {str(e)}")
            continue

        # Copy files
        for img in train:
            shutil.copy(
                os.path.join(expr_dir, img),
                os.path.join(TRAIN_DIR, expression, img)
            )
        for img in val:
            shutil.copy(
                os.path.join(expr_dir, img),
                os.path.join(VAL_DIR, expression, img)
            )
        for img in test:
            shutil.copy(
                os.path.join(expr_dir, img),
                os.path.join(TEST_DIR, expression, img)
            )

        print(f"✅ {expression}: {len(train)} train, {len(val)} val, {len(test)} test")

    print("\n🎉 Dataset splitting completed!")

if __name__ == "__main__":
    create_splits()