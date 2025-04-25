from .custom import *

import numpy as np
import pandas as pd
from monai.transforms import *
from src.data_exploration import explore

__all__ = [
    "train_transforms",
    "modality_transforms"
    "public_transforms",
    "valid_transforms",
    "tuning_transforms",
    "unlabeled_transforms",
]

train_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img", "label"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        RandZoomd(
            keys=["img", "label"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.5,
            mode=["area", "nearest"],
            keep_size=False,
        ),
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        # # >>> Intensity transforms
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
        EnsureTyped(keys=["img", "label"]),
    ]
)

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

        sample = explore.DataLabels(d[self.source_key])
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
            slices = self.cropper.compute_slices(roi_start=[top, left], roi_end=[bottom, right])
            for key in self.key_iterator(d):
                kwargs = {}
                if isinstance(self.cropper, LazyTrait):
                    kwargs["lazy"] = lazy_
                    d[key] = self.cropper(d[key], slices, **kwargs)  # type: ignore
            yield d
    
    def generate_bounds(self, meta_path):
        labels = explore.DataLabels(meta_path)
        for index in range(len(labels.df)):
            box = labels.df.iloc[index]
            yield [box['Left'], box['Top'], box['Right'], box['Bottom']]


modality_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img", "label"], image_only=True),
        EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        RandSmartCropSamplesd(keys=["img", "label"], source_key="meta", num_samples=1),
        Resized(keys=["img", "label"], spatial_size=(512, 512), mode=["area", "nearest-exact"]),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.75, spatial_axes=[0, 1]),
        #IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        # # >>> Intensity transforms
        #RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        #RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        #RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        #RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        #RandGaussianSharpend(keys=["img"], prob=0.25),
        EnsureTyped(keys=["img", "label"]),
    ]
)

public_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"], image_only=True),
        BoundaryExclusion(keys=["label"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        Rotate90d(k=1, keys=["label"], spatial_axes=(0, 1)),
        Flipd(keys=["label"], spatial_axis=0),
        EnsureTyped(keys=["img", "label"]),
    ]
)


valid_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"], allow_missing_keys=True, image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], allow_missing_keys=True, channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
    ]
)

tuning_transforms = Compose(
    [
        CustomLoadImaged(keys=["img"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
        ScaleIntensityd(keys=["img"]),
        EnsureTyped(keys=["img"]),
    ]
)

unlabeled_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
        RandZoomd(
            keys=["img"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.25,
            mode=["area"],
            keep_size=False,
        ),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img"], spatial_size=512),
        RandSpatialCropd(keys=["img"], roi_size=512, random_size=False),
        EnsureTyped(keys=["img"]),
    ]
)


def get_pred_transforms():
    """Prediction preprocessing"""
    pred_transforms = Compose(
        [
            # >>> Load and refine data
            CustomLoadImage(image_only=True),
            CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
            EnsureChannelFirst(channel_dim=-1),  # image: (3, H, W)
            ScaleIntensity(),
            EnsureType(data_type="tensor"),
        ]
    )

    return pred_transforms
