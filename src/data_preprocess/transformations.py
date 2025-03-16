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

def apply_tranformations(input_size, img_path, gt_path, target_path):
    os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)

    train_transforms = Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        SpatialPadd(keys=["img", "label"], spatial_size=input_size),
        RandSpatialCropd(keys=["img", "label"], roi_size=input_size, random_size=False),
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
        
        Image.fromarray((transformed_img * 255).astype(np.uint8)).save(os.path.join(target_path, "images", f"{img_name}.png"))
        Image.fromarray((transformed_label * 255).astype(np.uint8)).save(os.path.join(target_path, "labels", f"{img_name}.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply MONAI transformations.")
    parser.add_argument("--input_dir", default="../../data/preprocessing_outputs/data_science_bowl/normalized_data/images" , type=str, required=False, help="Path to input images.")
    parser.add_argument("--label_dir", default="../../data/preprocessing_outputs/data_science_bowl/normalized_data/labels", type=str, required=False, help="Path to label images.")
    parser.add_argument("--output_dir", default="../../data/preprocessing_outputs/data_science_bowl/transformed_images_labels" , type=str, required=False, help="Path to save transformed images.")
    parser.add_argument("--input_size", default=256 , type=int, required=False, help="Image size for transformation.")
    args = parser.parse_args()
    os.makedirs('../../data/preprocessing_outputs', exist_ok=True)
    apply_tranformations(args.input_size, args.input_dir, args.label_dir, args.output_dir)
    print("Preprocessing complete.")
