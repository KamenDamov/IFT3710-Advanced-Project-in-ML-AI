import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.morphology import label
join = os.path.join
from tqdm import tqdm

def convert_train_from_png(root, destination):
    # Define paths
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


def rle_decode(mask_rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint16)
    if pd.isna(mask_rle):
        return mask.reshape(shape)

    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    for i, (start, length) in enumerate(zip(starts, lengths), start=1):
        mask[start: start + length] = i

    return mask.reshape(shape)


def convert_test_from_csv(root, destination, csv_path):
    masks = pd.read_csv(csv_path)
    # Define paths
    image_output_dir = join(destination, "images")
    mask_output_dir = join(destination, "labels")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Get all image directories
    image_dirs = os.listdir(root)

    masks_dict = {}
    # Process each image
    for idx, row in masks.iterrows():
        image_id, encoded_pixels, height, width, _ = row

        # Decode RLE mask
        mask = rle_decode(encoded_pixels, (int(height), int(width)))

        # Label connected components to assign unique IDs
        labeled_mask = label(mask, connectivity=1)

        # Save mask as TIFF
        masks_dict[image_id] = labeled_mask

    # Process each image and its masks
    for idx, image_dir in tqdm(sorted(enumerate(image_dirs))):
        # Define new filenames
        new_basename = f"cell_sb_{idx + 1:05d}"
        new_image_path = join(image_output_dir, new_basename + ".jpg")
        new_mask_path = join(mask_output_dir, new_basename + "_label.tiff")

        source_image = join(root, image_dir, 'images')
        image = io.imread(join(source_image, os.listdir(source_image)[0]))

        # Remove alpha channel if present
        if image.shape[-1] == 4:  # RGBA
            image = color.rgba2rgb(image)  # Convert to RGB
            image = (image * 255).astype(np.uint8)  # Scale back to [0, 255]
        elif image.dtype == np.uint16:  # Convert 16-bit grayscale to 8-bit
            image = (image / 256).astype(np.uint8)  # Scale to 0-255

        io.imsave(new_image_path, image, check_contrast=False)

        io.imsave(new_mask_path, masks_dict[image_dir], check_contrast=False)

    print("Dataset reorganization complete!")


if __name__ == '__main__':
    root = "/home/ggenois/Downloads/datasciencebowl_dataset/stage1_train"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/dataScienceBowl_converted/train"
    convert_train_from_png(root, destination)
    root = "/home/ggenois/Downloads/datasciencebowl_dataset/stage1_test"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/dataScienceBowl_converted/test"
    csv_path = "/home/ggenois/Downloads/datasciencebowl_dataset/stage1_solution.csv"
    convert_test_from_csv(root, destination, csv_path)
    root = "/home/ggenois/Downloads/datasciencebowl_dataset/stage2_test_final"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/dataScienceBowl_converted/test_final"
    csv_path = "/home/ggenois/Downloads/datasciencebowl_dataset/stage2_solution_final.csv"
    convert_test_from_csv(root, destination, csv_path)