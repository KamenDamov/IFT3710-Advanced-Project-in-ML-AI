import os
import numpy as np
from skimage import io, color
join = os.path.join
from tqdm import tqdm

# Define paths
root = "/home/ggenois/Downloads/datasciencebowl_dataset/stage1_train"
destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/dataScienceBowl_converted"
image_output_dir = join(destination, "images")
mask_output_dir = join(destination, "labels")
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# Get all image directories
image_dirs = os.listdir(root)

# Process each image and its masks
for idx, image_dir in tqdm(sorted(enumerate(image_dirs))):
    # Define new filenames
    new_basename = f"cell_sb_{idx + 1:05d}"
    new_image_path = join(image_output_dir, new_basename + ".jpg")
    new_mask_path = join(mask_output_dir, new_basename + "_label.tiff")

    source_image = join(root, image_dir, 'images')
    source_masks = join(root, image_dir, 'masks')

    image = io.imread(join(source_image, os.listdir(source_image)[0]))

    # Remove alpha channel if present
    if image.shape[-1] == 4:  # RGBA
        image = color.rgba2rgb(image)  # Convert to RGB
        image = (image * 255).astype(np.uint8)  # Scale back to [0, 255]

    io.imsave(new_image_path, image, check_contrast=False)

    mask_paths = [join(source_masks, mask) for mask in os.listdir(source_masks)]
    # Stack masks with unique labels
    if mask_paths:
        combined_mask = np.zeros_like(io.imread(mask_paths[0]), dtype=np.uint16)

        for label, mask_path in enumerate(mask_paths, start=1):
            mask = io.imread(mask_path)
            combined_mask[mask > 0] = label  # Assign unique label

        io.imsave(new_mask_path, combined_mask, check_contrast=False)


    # print(f"Processed {image_dir} -> {new_image_path}, {new_mask_path}")

print("Dataset reorganization complete!")
