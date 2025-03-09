import os
join = os.path.join
import argparse
from skimage import io, segmentation, morphology, exposure, img_as_ubyte
import numpy as np
import tifffile as tif
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import monai
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
import traceback
import pickle



#%% Data Normalization

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

def normalization():
    source_path = '../../data/Tuning'
    target_path = '../../data/preprocessing_outputs/tuning/normalized_data'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Get images names
    images = join(source_path, 'images')
    labels = join(source_path, 'labels')

    img_names = sorted(os.listdir(images))
    gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]

    # Create directories for preprocessed images and ground truth
    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)

    log = ["Failed to process images: \n"]
    for img_name, gt_name in zip(tqdm(img_names, desc="Normalizing images"), gt_names):
        try:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(images, img_name))
            else:
                img_data = io.imread(join(images, img_name))
            gt_data = tif.imread(join(labels, gt_name))

            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
                
            # conver instance bask to three-class mask: interior, boundary
            interior_map = create_interior_map(gt_data.astype(np.int16))
            
            io.imsave(join(target_path, 'images', img_name.split('.')[0]+'.png'), pre_img_data.astype(np.uint8), check_contrast=False)
            io.imsave(join(target_path, 'labels', gt_name.split('.')[0]+'.png'), interior_map.astype(np.uint8), check_contrast=False)
        except: 
            log.append(img_name)      

    with open('logs.txt', 'a') as f: 
        f.write("\n".join(log))
        f.close()   


#%% Image Tranformations

def apply_tranformations():
    input_size = 256

    img_path = "../../data/preprocessing_outputs/tuning/normalized_data/images"
    gt_path = "../../data/preprocessing_outputs/tuning/normalized_data/labels"
    target_path = "../../data/preprocessing_outputs/tuning/transformed_images_labels"

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
    print("Preprocessing data...")
    os.makedirs('../../data/preprocessing_outputs', exist_ok=True)
    normalization()
    apply_tranformations()
    print("Preprocessing complete.")
