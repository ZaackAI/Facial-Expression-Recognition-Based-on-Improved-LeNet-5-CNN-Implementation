


#Occlusion test
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def show_occlusion_examples():
    test_dir = "data/test"
    for cls in os.listdir(test_dir):
        img_path = os.path.join(test_dir, cls, os.listdir(os.path.join(test_dir, cls))[0])
        img = np.array(Image.open(img_path))
        
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        cases = [None, "eye", "mouth", "left"]
        
        for i, occ_type in enumerate(cases):
            occluded = img.copy()
            if occ_type == "eye":
                occluded[8:18, 6:26] = 0  # Paper's exact eye region
            elif occ_type == "mouth":
                occluded[18:28, 8:24] = 0  # Paper's mouth region
            elif occ_type == "left":
                occluded[:, :16] = 0  # Left half
            
            axs[i].imshow(occluded, cmap='gray')
            axs[i].set_title(f"{occ_type or 'Clean'}")
            axs[i].axis('off')
        
        plt.suptitle(f"Class: {cls}")
        plt.show()

show_occlusion_examples()