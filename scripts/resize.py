import cv2
import os

input_dir = "data/CK+48"
output_dir = "data/CK+_resized"
os.makedirs(output_dir, exist_ok=True)

for expression in os.listdir(input_dir):
    expr_dir = os.path.join(input_dir, expression)
    if not os.path.isdir(expr_dir):
        continue
    
    output_expr_dir = os.path.join(output_dir, expression)
    os.makedirs(output_expr_dir, exist_ok=True)
    
    for img_name in os.listdir(expr_dir):
        img_path = os.path.join(expr_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (32, 32))  # Resize to 32×32
        cv2.imwrite(os.path.join(output_expr_dir, img_name), img_resized)

print("Resizing complete! Resized images saved to:", output_dir)