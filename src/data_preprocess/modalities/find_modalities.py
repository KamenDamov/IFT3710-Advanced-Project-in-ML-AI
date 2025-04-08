import pickle
from sklearn.cluster import KMeans
from glob import glob
import os
from monai.data import Dataset, DataLoader
import torch
import numpy as np
import multiprocessing
from tqdm import tqdm
from src.data_preprocess.modalities.train_tools.data_utils.transforms import modality_transforms, train_transforms
from src.data_preprocess.modalities.train_tools.models import MEDIARFormer
from src.data_exploration import explore
join = os.path.join

def generate_dataset(dataroot):
    # Load all image paths
    for name, df in explore.enumerate_frames(dataroot):
        if ".labels" in name:
            for index in range(len(df)):
                sample = explore.DataSample(dataroot, df.iloc[index])
                yield { "img": sample.normal_image, "label": sample.normal_mask, "meta": sample.meta_frame, "name": sample.name }

# Function to extract features from model
def extract_features(model, loader):
    features_list = []

    with torch.no_grad():
        for batch in tqdm(loader):
            img_tensor = batch["img"].to("cpu")
            features = model(img_tensor)
            features_list.append(features.cpu().numpy().flatten())  # Flatten for clustering

    return np.array(features_list)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    model_path1 = './models/mediar/pretrained/phase1.pth'
    weights1 = torch.load(model_path1, map_location="cpu")

    model = MEDIARFormer()
    model.load_state_dict(weights1, strict=False)

    # Create dictionary mapping image files to label files
    data_dicts = list(generate_dataset('./data'))

    dataset = Dataset(data=data_dicts, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    # Run feature extraction
    print("Extracting features...")
    features = extract_features(model, loader)
    print("Feature extraction complete!", features.shape)

    # Perform K-Means clustering
    print("Extracting modalities...")
    kmeans = KMeans(n_clusters=40, n_init=10, verbose=5)
    modalities = kmeans.fit_predict(features)
    print("Clustering complete!", kmeans.centers_)

    print("Saving modalities...")
    modalities_map = {}
    for idx, modality in tqdm(enumerate(modalities)):
        image_basename = os.path.splitext(os.path.basename(data_dicts[idx]['img']))
        if modality in modalities_map.keys():
            modalities_map[modality].append(image_basename)
        else:
            modalities_map[modality] = [image_basename]

    # Save dictionary to a pickle file
    pickle_file = "new_modalities.pkl"

    with open(pickle_file, "wb") as f:
        pickle.dump(modalities_map, f)

    print("Images sorted into modalities!")
