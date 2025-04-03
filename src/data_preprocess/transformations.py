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

def get_crop_size(images_paths):
    min_size = (np.inf, np.inf)
    selected_image = ""
    for _, image in enumerate(tqdm(images_paths, desc="getting min input size")):
        img = Image.open(f"{image}")
        width, high = img.size[0], img.size[1]
        if width < min_size[0] and high < min_size[1]:
            min_size = img.size
            selected_image = image
    return min_size, selected_image

def validate_mask(transformed_mask, crop_size):
    return ( sum(transformed_mask.flatten()) / (crop_size*crop_size) ) >= 0.20

def composed_transforms(crop_size):
    return Compose([
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

def apply_tranformations(crop_size, img_path, gt_path, target_path):
    os.makedirs(os.path.join(target_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)

    image_files = sorted([f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    data_dicts = [{"img": os.path.join(img_path, img), "label": os.path.join(gt_path, lbl), "name": os.path.splitext(img)[0]} for img, lbl in zip(image_files, label_files)]
    
    dataset = Dataset(data=data_dicts, transform=composed_transforms(crop_size))
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    batch_transform(loader, crop_size, target_path)
    
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
    #crop_size, _ = get_crop_size([d['img'] for d in data_dicts])
    #dataset = Dataset(data=data_dicts, transform=composed_transforms(crop_size))
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
        #SpatialPadd(keys=["img", "label"], spatial_size=crop_size),
        #RandSpatialCropd(keys=["img", "label"], roi_size=crop_size, random_size=False),
        RandSmartCropd(keys=["img", "label"], source_key="meta"),
        #CropForegroundd(keys=["img", "label"], source_key="meta", select_fn=lambda csv: pd.read_csv(csv)),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        #RandZoomd(keys=["img", "label"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=["area", "nearest"]),
        Resized(keys=["img", "label"], spatial_size=(256, 256), mode=["area", "nearest"]),
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
        slices = self.select_slices(d[self.source_key])
        for key in self.key_iterator(d):
            kwargs = {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
        return d
    
    def select_slices(self, meta_path):
        df = pd.read_csv(meta_path)
        width, height = df['Right'][0], df['Bottom'][0]
        weights = list(df['Area'].sum() / df['Area'])
        # sample an integer according to given weights
        choice = self.R.choice(len(df), p=weights)
        df = df.iloc[choice]
        left = self.R.randint(0, df['Left']+1)
        right = self.R.randint(df['Right']+1, width+1)
        top = self.R.randint(0, df['Top']+1)
        bottom = self.R.randint(df['Bottom']+1, height+1)
        slices = self.cropper.compute_slices(roi_start=[top, left], roi_end=[bottom, right])
        return slices

def main2(base_dir):
    parser = argparse.ArgumentParser(description="Apply transformations.")
    parser.add_argument("--input_dir", default=f"{base_dir}/preprocessing_outputs/normalized_data/images" , type=str, required=False, help="Path to input images.")
    parser.add_argument("--label_dir", default=f"{base_dir}/preprocessing_outputs/normalized_data/labels", type=str, required=False, help="Path to label images.")
    parser.add_argument("--output_dir", default=f"{base_dir}/preprocessing_outputs/transformed_images_labels" , type=str, required=False, help="Path to save transformed images.")
    args = parser.parse_args()
    crop_size, _ = get_crop_size(args.input_dir)
    print(f"Input size: {crop_size}")
    os.makedirs(f'{base_dir}/preprocessing_outputs', exist_ok=True)
    apply_tranformations(min(crop_size), args.input_dir, args.label_dir, args.output_dir)
    print("Preprocessing complete.")

if __name__ == '__main__':
    base_dir = "../../data"
    main(base_dir)

