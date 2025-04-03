import os
join = os.path.join
import torch
import argparse
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from monai.data import PILReader
from monai.transforms.traits import LazyTrait
from monai.transforms.croppad.array import Crop
from monai.transforms import (
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    ScaleIntensityd,
    SpatialPadd,
    RandSpatialCropd,
    CropForegroundd,
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
from ..data_exploration import explore
from . import normalization

def batch_transform(loader, target_path):
    for _, batch in enumerate(tqdm(loader, desc="Transforming images and labels")):
        img_name = batch["name"][0]
        transformed_img = batch["img"].squeeze().numpy().transpose(1, 2, 0)
        transformed_label = batch["label"].squeeze().numpy()
        Image.fromarray((transformed_img * 255).astype(np.uint8)).save(os.path.join(target_path, "images", f"{img_name}.png"))
        Image.fromarray((transformed_label * 255).astype(np.uint8)).save(os.path.join(target_path, "labels", f"{img_name}.png"))

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
            image_files = [normalization.target_file(norm_target + img_path, ".png") for img_path in df["Path"]]
            label_files = [normalization.target_file(norm_target + mask_path, ".png") for mask_path in df["Mask"]]
            object_files = [normalization.target_file(process_target + mask_path, ".csv") for mask_path in df["Mask"]]
            for img, lbl, meta in sorted(zip(image_files, label_files, object_files)):
                if "Tuning" in img:
                    yield {"img": img, "label": lbl, "meta": meta, "name": explore.split_filepath(img)[1]}

def smart_transforms():
    return Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        RandSmartCropd(keys=["img", "label"], source_key="meta"),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        Resized(keys=["img", "label"], spatial_size=(512, 512), mode=["area", "nearest-exact"]),
        EnsureTyped(keys=["img", "label"]),
    ])

class RandSmartCropd(Cropd, Randomizable):
    """
    Base class for random crop transform.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        cropper: random crop transform for the input image.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    backend = Crop.backend

    def __init__(self, keys, source_key, allow_missing_keys: bool = False, lazy: bool = False):
        cropper = Crop(lazy=lazy)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.source_key = source_key

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        super().set_random_state(seed, state)
        if isinstance(self.cropper, Randomizable):
            self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, img_size) -> None:
        if isinstance(self.cropper, Randomizable):
            self.cropper.randomize(img_size)

    def __call__(self, data, lazy: bool | None = None):
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.cropper, LazyTrait):
            raise ValueError(
                "'self.cropper' must inherit LazyTrait if lazy is True "
                f"'self.cropper' is of type({type(self.cropper)}"
            )
        slices = self.select_slices(data['img'].shape, d[self.source_key])
        for key in self.key_iterator(d):
            kwargs = {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
        return d
    
    def select_slices(self, dims, meta_path):
        df = pd.read_csv(meta_path)
        width, height = df['Right'][0], df['Bottom'][0]
        assert width == dims[1]
        assert height == dims[2]
        weights = df['Area'].sum() / df['Area']
        weights = list(weights / weights.sum())
        # sample an integer according to given weights
        choice = self.R.choice(len(df), p=weights)
        df = df.iloc[choice]
        left = self.R.randint(0, df['Left']+1)
        right = self.R.randint(df['Right'], width+1)
        top = self.R.randint(0, df['Top']+1)
        bottom = self.R.randint(df['Bottom'], height+1)
        slices = self.cropper.compute_slices(roi_start=[left, top], roi_end=[right, bottom])
        return slices

if __name__ == '__main__':
    base_dir = "../../data"
    main(base_dir)

