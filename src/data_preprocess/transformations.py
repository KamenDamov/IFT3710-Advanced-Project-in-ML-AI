import os
join = os.path.join
import torch
import argparse
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from monai.data import PILReader
from monai.transforms.traits import LazyTrait, MultiSampleTrait
from monai.transforms.croppad.array import Crop
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandAxisFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    Resized,
    RandZoomd,
    EnsureTyped,
    Compose,
    Randomizable,
    Cropd,
)
from monai.data import Dataset, DataLoader
from PIL import Image
import argparse
from src.data_exploration import explore
from src.data_preprocess.modalities.train_tools.data_utils.transforms import RandSmartCropSamplesd

def batch_transform(samples, loader):
    for sample, batch in zip(samples, tqdm(loader, desc="Transforming images and labels")):
        for index in range(len(batch["name"])):
            transformed_img = batch["img"][index].numpy().transpose(1, 2, 0)
            transformed_label = batch["label"][index].squeeze().numpy()
            explore.safely_process([], save_transform, overwrite=True)(transformed_img, sample.transform_image(index))
            explore.safely_process([], save_transform, overwrite=True)(transformed_label, sample.transform_mask(index))

def save_transform(transformed_img, target):
    Image.fromarray((transformed_img * 255).astype(np.uint8)).save(target)

def main(dataroot):
    samples = explore.DataSet(dataroot)
    data_dicts = [{"img": sample.normal_image, "label": sample.normal_mask, "meta": sample.meta_frame, "name": sample.name} for sample in samples]
    dataset = Dataset(data=data_dicts, transform=smart_transforms())
    loader = DataLoader(dataset, batch_size=1, num_workers=1)
    batch_transform(samples, loader)

def smart_transforms():
    return Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        EnsureChannelFirstd(channel_dim="no_channel", keys=["label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        RandSmartCropSamplesd(keys=["img", "label"], source_key="meta", num_samples=5),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.75, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        Resized(keys=["img", "label"], spatial_size=(512, 512), mode=["area", "nearest-exact"]),
        EnsureTyped(keys=["img", "label"]),
    ])

if __name__ == '__main__':
    main("./data")
