import numpy as np
from PIL import Image as Image
import os

def explore_images_labels(path):
    """
    Load an image from a given path.
    """
    cell_path = path + "\\images"
    label_path = path + "\\labels"
    cell_list = os.listdir(cell_path)
    labels_list = os.listdir(label_path)
    zipped = list(zip(cell_list, labels_list))
    for cell, label in zipped: 
        print(cell_path + "\\" + cell, label_path + "\\" + label)
        print("image: ", np.array(Image.open(cell_path + "\\" + cell)).shape, "label: ", np.array(Image.open(label_path + "\\" + label)).shape)
        break

def explore_labels(path): 
    labels_list = os.listdir(path)
    for label in labels_list: 
        print("Shape: ", np.array(Image.open(path+"\\"+label)).shape)
        break

cycle_gan_path = "data\\preprocessing_outputs\\unified_augmented_data\\cycle_gan"
transformed = "data\\preprocessing_outputs\\transformed_images_labels"

synthetic_labels = "src\\data_augmentation\\gans\\base_gan\\generated_samples"

explore_images_labels(cycle_gan_path)
explore_images_labels(transformed)

explore_labels(synthetic_labels)
