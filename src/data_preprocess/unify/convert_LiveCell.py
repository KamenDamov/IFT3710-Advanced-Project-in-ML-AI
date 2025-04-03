import os
import shutil
import numpy as np
import pandas as pd
import json
from skimage import io, color, measure, draw
from skimage.morphology import label
from glob import glob
from tqdm import tqdm

def main(root, destination, json_path):
    # Load JSON file
    with open(json_path) as json_file:
        data = json.load(json_file)

    # Define paths
    image_output_dir = os.path.join(destination, "images")
    mask_output_dir = os.path.join(destination, "labels")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Extract image metadata
    image_dict = {img["id"]: img["file_name"] for img in data["images"]}
    annotations = data["annotations"]

    # Process each image
    for idx, (image_id, file_name) in tqdm(enumerate(image_dict.items())):
        new_basename = f"cell_lc_{idx + 1:05d}"
        new_image_path = os.path.join(image_output_dir, new_basename + ".jpg")
        new_mask_path = os.path.join(mask_output_dir, new_basename + "_label.tiff")

        # Load and save image
        image = io.imread(os.path.join(root, file_name))
        if image.shape[-1] == 4:  # RGBA
            image = color.rgba2rgb(image)
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint16:  # Convert 16-bit grayscale to 8-bit
            image = (image / 256).astype(np.uint8)
        io.imsave(new_image_path, image, check_contrast=False)

        # Create an empty mask
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint16)

        # Process segmentation data
        annotation_idx = 1
        for ann in annotations:
            if ann["image_id"] == image_id:
                for seg in ann["segmentation"]:
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                    rr, cc = draw.polygon(poly[:, 1], poly[:, 0], shape=(height, width))
                    mask[rr, cc] = annotation_idx  # Assign unique label
                    annotation_idx += 1

        # Save mask as TIFF
        io.imsave(new_mask_path, mask, check_contrast=False)

    print("Dataset conversion complete!")



if __name__ == '__main__':
    root = "/home/ggenois/Downloads/livecell_dataset/images/livecell_train_val_images"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/liveCell_converted/train"
    json_path = "/home/ggenois/Downloads/livecell_dataset/livecell_coco_train.json"
    main(root, destination, json_path)

    root = "/home/ggenois/Downloads/livecell_dataset/images/livecell_train_val_images"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/liveCell_converted/val"
    json_path = "/home/ggenois/Downloads/livecell_dataset/livecell_coco_val.json"
    main(root, destination, json_path)

    root = "/home/ggenois/Downloads/livecell_dataset/images/livecell_test_images"
    destination = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/liveCell_converted/test"
    json_path = "/home/ggenois/Downloads/livecell_dataset/livecell_coco_test.json"
    main(root, destination, json_path)
