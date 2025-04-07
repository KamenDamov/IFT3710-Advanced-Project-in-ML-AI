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

def batch_transform(loader, target_path):
    for batch in tqdm(loader, desc="Transforming images and labels"):
        for index in range(len(batch["name"])):
            img_name = batch["name"][index]
            transformed_img = batch["img"][index].numpy().transpose(1, 2, 0)
            transformed_label = batch["label"][index].squeeze().numpy()
            Image.fromarray((transformed_img * 255).astype(np.uint8)).save(os.path.join(target_path, "images", f"{img_name}.{index}.png"))
            Image.fromarray((transformed_label * 255).astype(np.uint8)).save(os.path.join(target_path, "labels", f"{img_name}.{index}.png"))

def main(dataroot):
    target_path = f"{dataroot}/preprocessing_outputs/transformed_images_labels"
    os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)

    data_dicts = list(assemble_dataset(dataroot))
    dataset = Dataset(data=data_dicts, transform=smart_transforms())
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    batch_transform(loader, target_path)

def assemble_dataset(dataroot):
    process_target = f"{dataroot}/processed"
    norm_target = f"{dataroot}/preprocessing_outputs/normalized_data"
    for name, df in explore.enumerate_frames(dataroot):
        if ".labels" in name:
            image_files = [explore.target_file(norm_target + img_path, ".png") for img_path in df["Path"]]
            label_files = [explore.target_file(norm_target + mask_path, ".png") for mask_path in df["Mask"]]
            object_files = [explore.target_file(process_target + mask_path, ".csv") for mask_path in df["Mask"]]
            for img, lbl, meta in sorted(zip(image_files, label_files, object_files), key=lambda x: x[0]):
                if ("WSI" not in img):
                    yield {"img": img, "label": lbl, "meta": meta, "name": explore.split_filepath(img)[1]}

def smart_transforms():
    return Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        EnsureChannelFirstd(channel_dim="no_channel", keys=["label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        RandSmartCropSamplesd(keys=["img", "label"], source_key="meta", num_samples=1),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        Resized(keys=["img", "label"], spatial_size=(512, 512), mode=["area", "nearest-exact"]),
        EnsureTyped(keys=["img", "label"]),
    ])

class RandSmartCropSamplesd(Cropd, Randomizable, MultiSampleTrait):
    backend = Crop.backend

    def __init__(self, keys, source_key, num_samples:int = 1, allow_missing_keys: bool = False, lazy: bool = False):
        cropper = Crop(lazy=lazy)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.source_key = source_key
        self.num_samples = num_samples

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        super().set_random_state(seed, state)
        if isinstance(self.cropper, Randomizable):
            self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, img_size) -> None:
        if isinstance(self.cropper, Randomizable):
            self.cropper.randomize(img_size)

    def __call__(self, data, lazy: bool | None = None):
        return list(self.internalCrop(data, lazy) for _ in range(self.num_samples))
    
    def internalCrop(self, data, lazy: bool | None = None):
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.cropper, LazyTrait):
            raise ValueError(
                "'self.cropper' must inherit LazyTrait if lazy is True "
                f"'self.cropper' is of type({type(self.cropper)}"
            )

        sample = explore.DataSample(d[self.source_key])
        slices = sample.select_slices(self.R)
        for key in self.key_iterator(d):
            kwargs = {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
        return d
    
if __name__ == '__main__':
    main("./data")
