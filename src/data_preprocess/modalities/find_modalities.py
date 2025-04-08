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
join = os.path.join

partition = 0
batch_size = 100
image_folder = "./data/raw/zenodo/Training-labeled/images"
label_folder = "./data/raw/zenodo/Training-labeled/labels"
save_path = 'features_list.pkl'
accel_device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_dataset(dataroot):
    # Load all image paths
    for name, df in explore.enumerate_frames(dataroot):
        if ".labels" in name:
            for index in range(len(df)):
                sample = explore.DataSample(dataroot, df.iloc[index])
                yield { "img": sample.normal_image, "label": sample.normal_mask, "meta": sample.meta_frame, "name": sample.name }

def load_features(save_path):
    features = []
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            while True:
                try:
                    features.append(pickle.load(f))
                except EOFError:
                    break
    return features


def main(partition, batch_size):
    model_path1 = './models/mediar/pretrained/phase1.pth'
    weights1 = torch.load(model_path1, map_location=accel_device)

    model = MEDIARFormer()
    model.load_state_dict(weights1, strict=False)

    # Extract features for all images
    # Create dictionary mapping image files to label files
    data_dicts = list(generate_dataset("./data"))
    data_dicts = data_dicts[partition*batch_size:partition*batch_size+batch_size]

    dataset = Dataset(data=data_dicts, transform=modality_transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=1)
    dataset.cache_data = False

    # Function to extract features from model
    def extract_features(model, loader):
        model.to(accel_device)
        features_list = []
        model.eval()

        with torch.no_grad():
            for batch in tqdm(loader):
                img_tensor = batch["img"].to(accel_device)
                features = model(img_tensor)

                with open(save_path, mode="ab") as f:
                    pickle.dump(features.cpu().numpy().flatten(), f)
                # features_list.append(features.cpu().numpy().flatten())  # Flatten for clustering

                del batch, img_tensor, features
                torch.cuda.empty_cache()
                gc.collect()

        return np.array(features_list)

    # Run feature extraction
    print("Extracting features...")
    extract_features(model, loader)
    features = load_features(save_path)
    return features


def get_modalities(features, image_folder, label_folder):
    # Perform K-Means clustering
    print("Extracting modalities...")
    kmeans = KMeans(n_clusters=40)
    modalities = kmeans.fit_predict(features)

    # Load all image paths
    image_paths = glob(os.path.join(image_folder, "*"))
    label_paths = glob(os.path.join(label_folder, "*"))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=int, default=partition)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--do_get_modalities", type=str, default=False)
    parser.add_argument("--do_get_features", type=str, default=True)
    parser.add_argument("--image_folder", type=str, default=image_folder)
    parser.add_argument("--label_folder", type=str, default=label_folder)
    args = parser.parse_args()
    partition = args.partition
    batch_size = args.batch_size
    do_get_modalities = args.do_get_modalities
    do_get_features = args.do_get_features
    image_folder = args.image_folder
    label_folder = args.label_folder

    if do_get_features:
        features = main(partition, batch_size)
    else:
        features = load_features(save_path)

    print('Features extracted:', len(features))

    if do_get_modalities:
        get_modalities(features, image_folder, label_folder)
