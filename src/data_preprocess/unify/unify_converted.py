import os
import shutil
from glob import glob
from tqdm import tqdm

def main(root, destination):
    # Define paths
    image_output_dir = os.path.join(destination, "images")
    mask_output_dir = os.path.join(destination, "labels")
    os.makedirs(destination, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Walk through all subdirectories to find relevant files
    image_paths = glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    mask_paths = glob(os.path.join(root, "**", "*.tiff"), recursive=True)

    # Copy images
    for image_path in tqdm(image_paths):
        shutil.copy(image_path, image_output_dir)

    # Copy masks
    for mask_path in tqdm(mask_paths):
        shutil.copy(mask_path, mask_output_dir)

    print("Dataset unified successfully!")

if __name__ == '__main__':
    root = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/datasets_converted"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/unified_set"
    main(root, destination)
