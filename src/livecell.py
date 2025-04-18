import os
import numpy as np
import json
from skimage import draw
from tqdm import tqdm

from src.manage_files import *

###
# Download the LiveCell dataset from:
# https://github.com/sartorius-research/LIVECell
# http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip
# -- NOT -- https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data
#
# Raw file structure:
# /livecell
# ├── livecell_coco_test.json
# ├── livecell_coco_train.json
# ├── livecell_coco_val.json
# ├── images.zip                        ***
# ├── /images
# |   ├── /livecell_train_val_images
# |   └── /livecell_test_images
# └── /labels
#     ├── /livecell_train_val_images
#     └── /livecell_test_images
###

class LiveCellSet(BaseFileSet):
    def __init__(self):
        super().__init__("/livecell")
    
    def unpack(self, dataroot):
        super().unpack(dataroot)
        build_masks(dataroot + self.root)
        
    def mask_filepath(self, filepath):
        if "/images" in filepath:
            folder, name, ext = split_filepath(filepath)
            maskpath = folder.replace("/images", "/labels") + name + "_mask.tiff"
            yield (maskpath, MASK)

    def categorize(self, filepath):
        if "/labels" in filepath:
            return MASK
        if "/images" in filepath:
            return IMAGE

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
        for img in tqdm(data["images"], desc=f"Processing {targets}"):
            filepath, maskpath = sample_paths(root, targets, img["file_name"])
            safely_process([], mask_builder(data, img))(filepath, maskpath)

def mask_builder(data, img):
    def build(filepath, maskpath):
        # We don't need to load the actual image, the dimensions from the json file are correct
        segmentation = find_segmentations(data, img['id'])
        mask = build_mask(img['width'], img['height'], segmentation)
        save_image(maskpath, mask)
    return build

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
    root = "./data/raw"
    dataset = LiveCellSet()
    dataset.unpack(root)
    dataset.crosscheck(root)
