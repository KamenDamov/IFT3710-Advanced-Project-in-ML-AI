from matplotlib import pyplot as plt
from skimage import io, segmentation, morphology, exposure
import os
from os.path import join
import cv2
import numpy as np
import tifffile as tif

from tqdm import tqdm

source_path = "/home/ggenois/Downloads/neurips_dataset/Tuning"
target_path = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/dataset/train"


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


img_path = join(source_path, 'images')
gt_path =  join(source_path, 'labels')
img_names = sorted(os.listdir(img_path))
gt_names = [img_name.split('.')[0] + '_label.tiff' for img_name in img_names]

pre_img_path = join(target_path, 'images')
pre_gt_path = join(target_path, 'labels')
os.makedirs(pre_img_path, exist_ok=True)
os.makedirs(pre_gt_path, exist_ok=True)

for img_name, gt_name in zip(tqdm(img_names), gt_names):
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
        img_data = tif.imread(join(img_path, img_name))
    else:
        img_data = io.imread(join(img_path, img_name))
    gt_data = tif.imread(join(gt_path, gt_name))

    # normalize image data
    if len(img_data.shape) == 2:
        img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        img_data = img_data[:, :, :3]
    else:
        pass
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:, :, i]
        if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
            pre_img_data[:, :, i] = normalize_channel(img_channel_i, lower=1, upper=99)

    io.imsave(join(target_path, 'images', img_name.split('.')[0] + '.png'), pre_img_data.astype(np.uint8),
              check_contrast=False)

    # boundary_map = create_boundary_map(gt_data.astype(np.uint16))
    height, width = gt_data.shape
    # Convert mask to binary format if it's not
    mask = (gt_data > 0).astype(np.uint8)
    # Find contours (assuming white cells on black background)
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    interior_map = create_interior_map(gt_data)
    # contours = [np.argwhere(interior_map == 2)]
    contours, _ = cv2.findContours(np.uint8(interior_map==2)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    with open(join(target_path, 'labels', img_name.split('.')[0] + '.txt'), "w", encoding='utf-8') as f:
        for contour in contours:
            # Normalize polygon points to (0,1) range
            normalized_points = [(x / width, y / height) for x, y in contour.squeeze()]
            if len(normalized_points) < 3:  # Ignore non-polygon shapes
                continue
            # Convert to YOLO format
            yolo_line = f"0 " + " ".join([f"{x} {y}" for x, y in normalized_points]) + "\n"
            f.write(yolo_line)

