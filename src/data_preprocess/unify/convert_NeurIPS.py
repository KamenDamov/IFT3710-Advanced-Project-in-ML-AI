import os
import shutil
import numpy as np
from skimage import io, color
from glob import glob
from tqdm import tqdm

def main(root, destination):
    # Define paths
    image_output_dir = os.path.join(destination, "images")
    mask_output_dir = os.path.join(destination, "labels")
    os.makedirs(destination, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Find all images (.bmp, .png) and masks (.tif, .tiff)
    image_paths = glob(os.path.join(root, "images", "*.bmp")) + glob(os.path.join(root, "images", "*.png")) + glob(os.path.join(root, "images", "*.tif*"))
    mask_paths = glob(os.path.join(root, "labels", "*.tif")) + glob(os.path.join(root, "labels", "*.tiff"))

    # Map masks to their corresponding images
    mask_dict = {os.path.basename(p).replace("_label.tiff", "").replace("_label.tif", ""): p for p in mask_paths}

    # Process each image
    for idx, image_path in tqdm(sorted(enumerate(image_paths))):
        image_name = os.path.basename(image_path).rsplit(".", 1)[0]  # Get image name without extension
        if image_name not in mask_dict:
            print(f"Warning: No mask found for {image_name}, skipping.")
            continue

        mask_path = mask_dict[image_name]
        new_basename = f"cell_nr_{idx + 1:05d}"
        new_image_path = os.path.join(image_output_dir, new_basename + ".jpg")
        new_mask_path = os.path.join(mask_output_dir, new_basename + "_label.tiff")

        # Load and save image
        image = io.imread(image_path)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)  # Convert float to uint8
        elif image.dtype in [np.int16, np.int32]:
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.shape[-1] == 4:  # RGBA
            image = color.rgba2rgb(image)
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint16:  # Convert 16-bit grayscale to 8-bit
            image = (image / 256).astype(np.uint8)
        io.imsave(new_image_path, image, check_contrast=False)

        # Load and save mask
        mask = io.imread(mask_path)
        io.imsave(new_mask_path, mask, check_contrast=False)

    print("NeurIPS dataset conversion complete!")

if __name__ == '__main__':
    root = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/Training-labeled"
    destination ="/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/datasets_converted/neurIPS_converted"
    main(root, destination)