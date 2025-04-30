import os
import shutil
from sklearn.model_selection import train_test_split
from src.datasets.datasets import *

if __name__ == "__main__":
    target_base = "./src/NeurIPS-CellSeg/data/Train_Pre_3class"
    target_images = target_base + "/images/"
    target_labels = target_base + "/labels/"
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    dataset = list(DataSet("./data"))
    dataset = [sample for sample in dataset if "/Testing/Public" not in sample.df["Path"]]
    #dataset, test = train_test_split(dataset, test_size=0.2, random_state=42)

    for index, sample in enumerate(tqdm(dataset, desc="Copying images")):
        image_name = f"cell_{index:05d}.png"
        shutil.copyfile(sample.normal_image, target_images + image_name)

    for index, sample in enumerate(tqdm(dataset, desc="Copying labels")):
        label_name = f"cell_{index:05d}_label.png"
        shutil.copyfile(sample.normal_mask, target_labels + label_name)
