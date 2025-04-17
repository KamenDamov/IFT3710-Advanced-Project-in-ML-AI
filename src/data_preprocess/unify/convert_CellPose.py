import os
import shutil
import numpy as np
import json
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

    # Find all image-mask pairs in the root directory
    image_paths = sorted(glob(os.path.join(root, "*_img.png")))  # Assuming images are PNGs
    mask_paths = sorted(glob(os.path.join(root, "*_masks.png")))  # Assuming masks have '_masks' suffix

    # Ensure corresponding images and masks match
    image_dict = {os.path.basename(p).replace("_img.png", ""): p for p in image_paths}
    mask_dict = {os.path.basename(p).replace("_masks.png", ""): p for p in mask_paths}

    # Process each pair
    for idx, (key, image_path) in tqdm(sorted(enumerate(image_dict.items()))):
        if key not in mask_dict:
            print(f"Warning: No mask found for {key}, skipping.")
            continue

        mask_path = mask_dict[key]
        new_basename = f"cell_cp_{idx + 1:05d}"
        new_image_path = os.path.join(image_output_dir, new_basename + ".jpg")
        new_mask_path = os.path.join(mask_output_dir, new_basename + "_label.tiff")

        # Load and save image
        image = io.imread(image_path)
        if image.shape[-1] == 4:  # RGBA
            image = color.rgba2rgb(image)
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint16:  # Convert 16-bit grayscale to 8-bit
            image = (image / 256).astype(np.uint8)
        io.imsave(new_image_path, image, check_contrast=False)

        # Load and save mask
        mask = io.imread(mask_path)
        if mask[mask == (2 ** 16 - 1)].any():
            print("Warning: Mask contains 16-bit values, converting to 8-bit.")
        io.imsave(new_mask_path, mask, check_contrast=False)

    print("Cellpose dataset conversion complete!")

if __name__ == '__main__':
    root = "./data/raw/cellpose/train"
    destination = "./data/unify/cellpose/train"
    main(root, destination)

    root = "./data/raw/cellpose/train_cyto2"
    destination = "./data/unify/cellpose/train_cyto2"
    main(root, destination)

    root = "./data/raw/cellpose/test"
    destination = "./data/unify/cellpose/test"
    main(root, destination)
