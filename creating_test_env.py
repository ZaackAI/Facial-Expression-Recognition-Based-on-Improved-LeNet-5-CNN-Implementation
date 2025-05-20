import os
import shutil
from sklearn.model_selection import train_test_split

# Configuration
SOURCE_DIR = "data/CK+_resized"
TEST_DIR = "data/test"
VAL_DIR = "data/val"
TRAIN_DIR = "data/train"
TEST_RATIO = 0.2  # 20% test
VAL_RATIO = 0.25  # 25% of remaining -> 20% of total
SEED = 42

# Clean existing directories
for dir_path in [TEST_DIR, VAL_DIR, TRAIN_DIR]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# Create splits per class
for expression in os.listdir(SOURCE_DIR):
    expr_dir = os.path.join(SOURCE_DIR, expression)
    if not os.path.isdir(expr_dir):
        continue

    # Create class subdirectories
    for dir_path in [TEST_DIR, VAL_DIR, TRAIN_DIR]:
        os.makedirs(os.path.join(dir_path, expression), exist_ok=True)

    # Get and shuffle images
    images = [f for f in os.listdir(expr_dir) if f.endswith(('.png', '.jpg'))]
    
    # First split: 80% train+val, 20% test
    trainval_images, test_images = train_test_split(
        images,
        test_size=TEST_RATIO,
        random_state=SEED
    )
    
    # Second split: 75% train, 25% val (of remaining 80%)
    train_images, val_images = train_test_split(
        trainval_images,
        test_size=VAL_RATIO,
        random_state=SEED
    )

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(expr_dir, img), 
                   os.path.join(TRAIN_DIR, expression, img))
    
    for img in val_images:
        shutil.copy(os.path.join(expr_dir, img),
                   os.path.join(VAL_DIR, expression, img))
        
    for img in test_images:
        shutil.copy(os.path.join(expr_dir, img),
                   os.path.join(TEST_DIR, expression, img))

print(f"""
✅ Correct splits created:
- Train: {TRAIN_DIR} (~60%)
- Val: {VAL_DIR} (~20%)
- Test: {TEST_DIR} (~20%)
""")

# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # Paths
# source_dir = "data/CK+_resized"
# test_dir = "data/test"
# os.makedirs(test_dir, exist_ok=True)

# # Split ratio: 20% test, 80% unused (or reuse for validation)
# test_size = 0.2

# for expression in os.listdir(source_dir):
#     expr_dir = os.path.join(source_dir, expression)
#     if not os.path.isdir(expr_dir):
#         continue

#     # List all images in the class
#     images = [f for f in os.listdir(expr_dir) if f.endswith(('.png', '.jpg'))]
    
#     # Split into test (20%) and unused (80%)
#     test_images, _ = train_test_split(images, test_size=1-test_size, random_state=42)

#     # Copy test images
#     os.makedirs(os.path.join(test_dir, expression), exist_ok=True)
#     for img in test_images:
#         src = os.path.join(expr_dir, img)
#         dst = os.path.join(test_dir, expression, img)
#         shutil.copy(src, dst)

# print(f"Test set created at {test_dir}")