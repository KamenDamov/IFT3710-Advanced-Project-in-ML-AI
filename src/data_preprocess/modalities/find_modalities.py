import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from glob import glob
import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import tifffile as tif
from monai.data import Dataset, DataLoader
import torch
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from train_tools.data_utils.transforms import train_transforms
# from train_tools import *
from train_tools.models import MEDIARFormer
# from core.MEDIAR import Predictor, EnsemblePredictor


model_path1 = 'phase1.pth'
weights1 = torch.load(model_path1, map_location="cpu")

model = MEDIARFormer()
model.load_state_dict(weights1, strict=False)


# Function to pad image to the nearest multiple of 32
def pad_image(image):
    c, h, w = image.shape
    new_h = (h + 31) // 32 * 32  # Round up to the nearest multiple of 32
    new_w = (w + 31) // 32 * 32
    padded_image = np.zeros((c, new_h, new_w), dtype=image.dtype)
    padded_image[:, :h, :w] = image  # Copy original image content
    return padded_image



# Load all image paths
image_folder = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/Training-labeled/images"
image_paths = glob(os.path.join(image_folder, "*"))
label_folder = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/Training-labeled/labels"
label_paths = glob(os.path.join(label_folder, "*"))

# Extract features for all images
data_dicts = [{"img": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]

dataset = Dataset(data=data_dicts, transform=train_transforms)
loader = DataLoader(dataset, batch_size=1, num_workers=1)


# Function to extract features from model
def extract_features(model, loader):
    features_list = []

    with torch.no_grad():
        for batch in loader:
            img_tensor = batch["img"].to("cpu")
            features = model(img_tensor)
            features_list.append(features.cpu().numpy().flatten())  # Flatten for clustering

    return np.array(features_list)


# Run feature extraction
features = extract_features(model, loader)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=40)
labels = kmeans.fit_predict(features)

# Map images to clusters
image_clusters = {img: label for img, label in zip(image_paths, labels)}


output_root = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/modalities"
os.makedirs(output_root, exist_ok=True)

for img_path, modality in image_clusters.items():
    modality_folder = os.path.join(output_root, modality)
    os.makedirs(modality_folder, exist_ok=True)

    shutil.move(img_path, os.path.join(modality_folder, os.path.basename(img_path)))

print("Images sorted into modalities!")
