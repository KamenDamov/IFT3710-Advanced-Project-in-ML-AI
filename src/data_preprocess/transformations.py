import os
join = os.path.join
import argparse
import numpy as np
from tqdm import tqdm
from monai.data import PILReader
from monai.transforms import (
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    ScaleIntensityd,
    SpatialPadd,
    RandSpatialCropd,
    RandAxisFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    RandZoomd,
    EnsureTyped,
    Compose,
)
from monai.data import Dataset, DataLoader
from PIL import Image
import argparse

def get_crop_size(images_path):
    images = os.listdir(images_path)
    min_size = (np.inf, np.inf)
    selected_image = ""
    for _, image in enumerate(tqdm(images, desc="getting min input size")):
        img = Image.open(f"{images_path}/{image}")
        width, high = img.size[0], img.size[1]
        if width < min_size[0] and high < min_size[1]:
            min_size = img.size
            selected_image = image
    return min_size, selected_image

def validate_mask(transformed_mask, crop_size):
    return ( sum(transformed_mask.flatten()) / (crop_size*crop_size) ) >= 0.5

def apply_tranformations(crop_size, img_path, gt_path, target_path):
    os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)

    train_transforms = Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        SpatialPadd(keys=["img", "label"], spatial_size=crop_size),
        RandSpatialCropd(keys=["img", "label"], roi_size=crop_size, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandZoomd(keys=["img", "label"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=["area", "nearest"]),
        EnsureTyped(keys=["img", "label"]),
    ])
    
    image_files = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    data_dicts = [{"img": os.path.join(img_path, img), "label": os.path.join(gt_path, lbl), "name": os.path.splitext(img)[0]} for img, lbl in zip(image_files, label_files)]
    
    dataset = Dataset(data=data_dicts, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    
    for _, batch in enumerate(tqdm(loader, desc="Transforming images and labels")):
        img_name = batch["name"][0]
        transformed_img = batch["img"].squeeze().numpy().transpose(1, 2, 0)
        transformed_label = batch["label"].squeeze().numpy()
        if validate_mask(transformed_label, crop_size):
            Image.fromarray((transformed_img * 255).astype(np.uint8)).save(os.path.join(target_path, "images", f"{img_name}.png"))
            Image.fromarray((transformed_label * 255).astype(np.uint8)).save(os.path.join(target_path, "labels", f"{img_name}.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply transformations.")
    parser.add_argument("--input_dir", default="C:/Users/Samir/Documents/GitHub/IFT3710-Advanced-Project-in-ML-AI/data/preprocessing_outputs/normalized_data/images" , type=str, required=False, help="Path to input images.")
    parser.add_argument("--label_dir", default="C:/Users/Samir/Documents/GitHub/IFT3710-Advanced-Project-in-ML-AI/data/preprocessing_outputs/normalized_data/labels", type=str, required=False, help="Path to label images.")
    parser.add_argument("--output_dir", default="C:/Users/Samir/Documents/GitHub/IFT3710-Advanced-Project-in-ML-AI/data/preprocessing_outputs/transformed_images_labels" , type=str, required=False, help="Path to save transformed images.")
    args = parser.parse_args()
    crop_size, _ = get_crop_size(args.input_dir)
    print(f"Input size: {crop_size}")
    os.makedirs('../../data/preprocessing_outputs', exist_ok=True)
    apply_tranformations(min(crop_size), args.input_dir, args.label_dir, args.output_dir)
    print("Preprocessing complete.")

