import gc
import pickle
from sklearn.cluster import KMeans
from glob import glob
import os
import argparse
from monai.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
from src.data_preprocess.modalities.train_tools.data_utils.transforms import train_transforms, modality_transforms
from src.data_preprocess.modalities.train_tools.models import MEDIARFormer
from src.data_exploration import explore
from src.datasets.datasets import DataSet
join = os.path.join

save_path = './data/features.pkl'
accel_device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_dataset(dataset):
    # Load all image paths
    for sample in dataset:
        yield { "img": sample.normal_image, "pickle": sample.embedding, "name": sample.name }
    for sample in dataset.unlabeled():
        yield { "img": sample.normal_image, "pickle": sample.embedding, "name": sample.name }

def load_embeddings(dataset):
    for sample in generate_dataset(dataset):
        if os.path.exists(sample["pickle"]):
            embedding = load_embedding(sample["pickle"])
            if embedding is not None:
                yield embedding

def load_embedding(target):
    try:
        with open(target, "rb") as f:
            return pickle.load(f)
    except:
        print(f"Failed to load: {target}")


# Function to extract features from model
def extract_features(model, loader):
    print("Extracting features...")
    model.to(accel_device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            target_path = batch["pickle"][0]
            img_tensor = batch["img"].to(accel_device)
            embedding = model(img_tensor)
            embedding = embedding.cpu().numpy().flatten()  # Flatten for clustering
            with open(target_path, mode="wb") as f:
                pickle.dump(embedding, f)
            # features_list.append(embedding)  # Flatten for clustering
            del batch, img_tensor, embedding
            torch.cuda.empty_cache()
            gc.collect()


def calculate_embeddings(dataset):
    model_path1 = './models/mediar/pretrained/phase1.pth'
    weights1 = torch.load(model_path1, map_location=accel_device)

    model = MEDIARFormer()
    model.load_state_dict(weights1, strict=False)

    # Extract features for all images
    # Create dictionary mapping image files to label files
    data_dicts = list(generate_dataset(dataset))
    data_dicts = [data for data in data_dicts if not os.path.exists(data['pickle'])]

    dataset = Dataset(data=data_dicts, transform=modality_transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)
    dataset.cache_data = False

    # Run feature extraction
    extract_features(model, loader)


def computer_clusters(features):
    # Perform K-Means clustering
    n_clusters=40
    print(f"Extracting modalities... {n_clusters} clusters from {len(features)} samples")
    clustering = KMeans(n_clusters=n_clusters, random_state=0, verbose=5)
    clustering.fit(features)
    
    print("Saving cluster centers...")
    # Save cluster centers to a pickle file
    with open(save_path, "wb") as f:
        pickle.dump(clustering.cluster_centers_, f)

    print("Images sorted into modalities!")
    return clustering

def load_clusters(save_path):
    with open(save_path, "rb") as f:
        clusters = pickle.load(f)
    clustering = KMeans(n_clusters=clusters.shape[0], init=clusters, n_init=1, random_state=0, verbose=5)
    clustering.fit(clusters)
    clustering.cluster_centers_ = clusters
    return clustering

if __name__ == '__main__':
    dataset = DataSet("./data")
    if not os.path.exists(save_path):
        calculate_embeddings(dataset)
        features = list(load_embeddings(dataset))
        clustering = computer_clusters(features)
    else:
        clustering = load_clusters(save_path)
        features = list(load_embeddings(dataset))
    modalities = clustering.predict(features)
    print(modalities)
