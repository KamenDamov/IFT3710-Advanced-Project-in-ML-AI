import os.path
from skimage import io, segmentation, morphology, exposure
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from glob import glob

source_path = "/home/ggenois/Downloads/neurips_dataset/Tuning"
target_path = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/dataset/train"

# Load an image and its corresponding mask
image_path = os.path.join(source_path, "images", "cell_00001.tiff")
mask_path = os.path.join(source_path, "labels", "cell_00001_label.tiff")
yolo_mask_path = os.path.join(target_path, "labels", "cell_00001.txt")


image = tiff.imread(image_path)
mask = tiff.imread(mask_path)

# Convert mask to RGB overlay
mask_colored = np.zeros_like(image)
mask_colored[mask > 0] = [255, 0, 0]  # Red mask

# Overlay mask on image
overlay = cv2.addWeighted(image, 0.1, mask_colored, 0.2, 0)

# Display image and mask
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Image with Mask Overlay of Whole Cells from Tiff")
plt.imshow(overlay)
plt.show()

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

interior_map = create_interior_map(mask)
plt.figure(figsize=(10,5))
plt.title("Image with Mask Overlay Boundaries from Tiff")
plt.imshow(interior_map==2)
plt.show()


def draw_yolo_labels(image_path, label_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        points = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
        points = (points * np.array([w, h])).astype(int)  # Convert to image scale

        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    plt.figure(figsize=(10, 5))
    plt.title("YOLO Annotations")
    plt.imshow(image)
    plt.show()

# Example usage
draw_yolo_labels(image_path, yolo_mask_path)

