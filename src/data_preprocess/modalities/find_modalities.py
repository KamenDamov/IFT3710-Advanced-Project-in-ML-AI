import pickle
from sklearn.cluster import KMeans
from glob import glob
import os
from monai.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
from train_tools.data_utils.transforms import train_transforms
from train_tools.models import MEDIARFormer
join = os.path.join

model_path1 = 'phase1.pth'
weights1 = torch.load(model_path1, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
model = MEDIARFormer()
model.load_state_dict(weights1)

# Load all image paths
image_folder = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\unified_set\\images"
image_paths = glob(os.path.join(image_folder, "*"))
label_folder = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\unified_set\\labels"
label_paths = glob(os.path.join(label_folder, "*"))

# Extract features for all images
image_files = {f.split(".")[0]: f for f in os.listdir(image_folder) if f.endswith(".png")}
label_files = {f.split(".")[0]: f for f in os.listdir(label_folder) if f.endswith(".png")}

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
