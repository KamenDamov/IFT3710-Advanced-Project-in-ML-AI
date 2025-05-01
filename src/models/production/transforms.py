import numpy as np
import pandas as pd
from monai.transforms import *
from monai.data import PILReader
from src.datasets import datasets

def baseline_train_transforms(input_size):
    return Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint16),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            SpatialPadd(keys=["img", "label"], spatial_size=input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=input_size, random_size=False
            ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=0.15,
                min_zoom=0.8,
                max_zoom=1.5,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

val_transforms = Compose(
    [
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint16),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        # AsDiscreted(keys=['label'], to_onehot=3),
        EnsureTyped(keys=["img", "label"]),
    ]
)

post_transform_transforms = Compose(
    [
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint16),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.75, spatial_axes=[0, 1]),
        # # intensity transform
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        EnsureTyped(keys=["img", "label"]),
    ]
)

class RandSmartCropd(Cropd, Randomizable):
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

        sample = datasets.DataLabels(d[self.source_key])
        cropping = sample.select_slices(self.R)
        slices = self.cropper.compute_slices(roi_start=[cropping['Left'], cropping['Top']], roi_end=[cropping['Right'], cropping['Bottom']])
        for key in self.key_iterator(d):
            kwargs = {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
        return d

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

        sample = datasets.DataLabels(d[self.source_key])
        cropping = sample.select_slices(self.R)
        slices = self.cropper.compute_slices(roi_start=[cropping['Left'], cropping['Top']], roi_end=[cropping['Right'], cropping['Bottom']])
        for key in self.key_iterator(d):
            kwargs = {}
            if isinstance(self.cropper, LazyTrait):
                kwargs["lazy"] = lazy_
            d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
        return d

class EnumerateObjectCropd(Cropd, MultiSampleTrait):
    backend = Crop.backend

    def __init__(self, keys, source_key, allow_missing_keys: bool = False, lazy: bool = False):
        cropper = Crop(lazy=lazy)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.source_key = source_key

    def __call__(self, data, lazy: bool | None = None):
        return list(self.generate_crops(data, lazy))
    
    def generate_crops(self, data, lazy: bool | None = None):
        lazy_ = self.lazy if lazy is None else lazy
        if lazy_ is True and not isinstance(self.cropper, LazyTrait):
            raise ValueError(
                "'self.cropper' must inherit LazyTrait if lazy is True "
                f"'self.cropper' is of type({type(self.cropper)}"
            )
        for [left, top, right, bottom] in self.generate_bounds(data[self.source_key]):
            d = dict(data)
            d['box'] = [left, top, right, bottom]
            slices = self.cropper.compute_slices(roi_start=[left, top], roi_end=[right, bottom])
            for key in self.key_iterator(d):
                kwargs = {}
                if isinstance(self.cropper, LazyTrait):
                    kwargs["lazy"] = lazy_
                    d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
            yield d
    
    def generate_bounds(self, meta_path):
        labels = datasets.DataLabels(meta_path)
        for index in range(len(labels.df)):
            box = labels.df.iloc[index]
            yield [box['Left'], box['Top'], box['Right'], box['Bottom']]

smart_train_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint16),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        # >>> Spatial transforms
        RandSmartCropd(keys=["img", "label"], source_key="meta"),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.75, spatial_axes=[0, 1]),
        #IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        # # >>> Intensity transforms
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
        Resized(keys=["img"], spatial_size=(256, 256), mode=["area"]),
        EnsureTyped(keys=["img", "label"]),
    ]
)
