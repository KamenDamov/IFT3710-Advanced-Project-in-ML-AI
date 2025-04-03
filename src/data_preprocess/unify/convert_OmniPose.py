import os
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

    # Walk through all subdirectories to find relevant files
    image_paths = []
    mask_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if filename.endswith(".tif") and not filename.endswith("_masks.tif") and not filename.endswith("flows.tif"):
                image_paths.append(full_path)
            elif filename.endswith("_masks.tif"):
                mask_paths.append(full_path)

    # Ensure corresponding images and masks match
    image_dict = {os.path.basename(p).replace(".tif", ""): p for p in image_paths}
    mask_dict = {os.path.basename(p).replace("_masks.tif", ""): p for p in mask_paths}

    # Process each pair
    for idx, (key, image_path) in tqdm(sorted(enumerate(image_dict.items()))):
        if key not in mask_dict:
            print(f"Warning: No mask found for {key}, skipping.")
            continue

        mask_path = mask_dict[key]
        new_basename = f"cell_op_{idx + 1:05d}"
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
        io.imsave(new_mask_path, mask, check_contrast=False)

    print("Omnipose dataset conversion complete!")

if __name__ == '__main__':
    root = "/home/ggenois/Downloads/omnipose_dataset/datasets/bact_fluor/train_sorted"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/omniPose_converted/bact_fluor/train"
    main(root, destination)

    root = "/home/ggenois/Downloads/omnipose_dataset/datasets/bact_fluor/test_sorted"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/omniPose_converted/bact_fluor/test"
    main(root, destination)

    root = "/home/ggenois/Downloads/omnipose_dataset/datasets/bact_phase/train_sorted"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/omniPose_converted/bact_phase/train"
    main(root, destination)

    root = "/home/ggenois/Downloads/omnipose_dataset/datasets/bact_phase/test_sorted"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/omniPose_converted/bact_phase/test"
    main(root, destination)

