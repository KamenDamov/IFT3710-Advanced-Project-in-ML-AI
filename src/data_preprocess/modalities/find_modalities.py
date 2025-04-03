import pickle

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

from train_tools.data_utils.transforms import get_pred_transforms
from train_tools.data_utils.transforms import train_transforms, public_transforms
# from train_tools import *
from train_tools.models import MEDIARFormer
# from core.MEDIAR import Predictor, EnsemblePredictor
join = os.path.join


model_path1 = 'src/data_preprocess/modalities/phase1.pth'
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
image_folder = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/preprocessing_outputs/normalized_data/images"
image_paths = glob(os.path.join(image_folder, "*"))
label_folder = "/home/ggenois/PycharmProjects/IFT3710-Advanced-Project-in-ML-AI/data/preprocessing_outputs/normalized_data/labels"
label_paths = glob(os.path.join(label_folder, "*"))

# Extract features for all images
# data_dicts = [{"img": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]

image_files = {f.split(".")[0]: f for f in os.listdir(image_folder) if f.endswith(".png")}
label_files = {f.split("_label.")[0]: f for f in os.listdir(label_folder) if f.endswith(".png")}

# Create dictionary mapping image files to label files
data_dicts = [{'img': join(image_folder, img_file), 'label': join(label_folder, label_files[img_name])} for img_name, img_file in image_files.items() if img_name in label_files]

dataset = Dataset(data=data_dicts, transform=train_transforms)
loader = DataLoader(dataset, batch_size=1, num_workers=1)

# Function to extract features from model
def extract_features(model, loader):
    features_list = []

    with torch.no_grad():
        for batch in tqdm(loader):
            img_tensor = batch["img"].to("cpu")
            features = model(img_tensor)
            features_list.append(features.cpu().numpy().flatten())  # Flatten for clustering

    return np.array(features_list)


# Run feature extraction
print("Extracting features...")
features = extract_features(model, loader)

# Perform K-Means clustering
print("Extracting modalities...")
kmeans = KMeans(n_clusters=40)
modalities = kmeans.fit_predict(features)

print("Saving modalities...")
modalities_map = {}
for idx, modality in tqdm(enumerate(modalities)):
    image_basename = os.path.splitext(os.path.basename(image_paths[idx]))[0]
    if modality in modalities_map.keys():
        modalities_map[modality].append(image_basename)
    else:
        modalities_map[modality] = [image_basename]
print(modalities_map)

# Save dictionary to a pickle file
pickle_file = "new_modalities.pkl"

with open(pickle_file, "wb") as f:
    pickle.dump(modalities_map, f)

print("Images sorted into modalities!")
