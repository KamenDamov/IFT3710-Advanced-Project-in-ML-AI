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
    os.makedirs(destination, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Extract image metadata
    image_dict = {img["id"]: img["file_name"] for img in data["images"]}
    annotations = data["annotations"]
    
    # Process each image
    for idx, (image_id, file_name) in tqdm(sorted(enumerate(image_dict.items()))):
        new_basename = f"cell_lc_{idx + 1:05d}"
        new_image_path = os.path.join(image_output_dir, new_basename + ".jpg")
        new_mask_path = os.path.join(mask_output_dir, new_basename + "_label.tiff")

        if os.path.exists(new_image_path) and os.path.exists(new_mask_path):
            continue

        print(f"Processing {image_id}...", file_name)

        # Load and save image
        #prefix = file_name.split("_")[0]
        image = io.imread(os.path.join(root, file_name))
        if image.shape[-1] == 4:  # RGBA
            image = color.rgba2rgb(image)
            image = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint16:  # Convert 16-bit grayscale to 8-bit
            image = (image / 256).astype(np.uint8)
        io.imsave(new_image_path, image, check_contrast=False)

        mask = build_mask(annotations, image_id, image)
        # Save mask as TIFF
        io.imsave(new_mask_path, mask, check_contrast=False)

    print("Dataset conversion complete!")

def enumerate_annotations(root):
    for filepath in os.listdir(root):
        if filepath.endswith(".json"):
            with open(os.path.join(root, filepath)) as json_file:
                yield json.load(json_file), annotation_targets(filepath)

def annotation_targets(ann_file):
    if "train" in ann_file or "val" in ann_file:
        return "livecell_train_val_images"
    elif "test" in ann_file:
        return "livecell_test_images"

def build_masks(root):
    for data, targets in enumerate_annotations(root):
        for img in tqdm(data["images"]):
            filepath, maskpath = sample_paths(root, targets, img["file_name"])
            if os.path.exists(maskpath):
                continue
            image = io.imread(filepath)
            assert image.shape[0] == img['height'] and image.shape[1] == img['width']
            segmentation = find_segmentations(data, img['id'])
            mask = build_mask(img['width'], img['height'], segmentation)
            os.makedirs(os.path.dirname(maskpath), exist_ok=True)
            io.imsave(maskpath, mask, check_contrast=False)

def sample_paths(root, targets, file_name):
    file_name, ext = os.path.splitext(file_name)
    filepath = os.path.join(root, "images", targets, file_name + ext)
    maskpath = os.path.join(root, "labels", targets, f"{file_name}_mask.tiff")
    return filepath, maskpath

def find_segmentations(data, image_id):
    for ann in data["annotations"]:
        if ann["image_id"] == image_id:
            yield ann["segmentation"]

def build_mask(width, height, segmentation):
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint16)
    # Process segmentation data for each cell, 0 is background
    for idx, cell_seg in enumerate(segmentation, start=1):
        poly = np.array(cell_seg, dtype=np.int32).reshape((-1, 2))
        rr, cc = draw.polygon(poly[:, 1], poly[:, 0], shape=(height, width))
        mask[rr, cc] = idx  # Assign unique label
    return mask

if __name__ == '__main__':
    root = "./data/raw/livecell"
    build_masks(root)

if False and __name__ == '__main__':
    root = "./data/raw/livecell/images/livecell_train_val_images"
    destination = "./data/unify/livecell/images/livecell_train_val_images"
    json_path = "./data/raw/livecell/livecell_coco_train.json"
    main(root, destination, json_path)

if False and __name__ == '__main__':
    root = "./data/raw/livecell/images/livecell_train_val_images"
    destination = "./data/unify/livecell/images/livecell_train_val_images"
    json_path = "./data/raw/livecell/livecell_coco_val.json"
    main(root, destination, json_path)

if False and __name__ == '__main__':
    root = "./data/raw/livecell/images/livecell_test_images"
    destination = "/data/unify/livecell/images/livecell_test_images"
    json_path = "./data/raw/livecell/livecell_coco_test.json"
    main(root, destination, json_path)
