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
    for cell, label in zipped[-1:]: 
        print(cell_path + "\\" + cell, label_path + "\\" + label)
        image_tensor = np.array(Image.open(cell_path + "\\" + cell))
        label_tensor = np.array(Image.open(label_path + "\\" + label))
        print("image: ", image_tensor.shape, "label: ", label_tensor.shape)

def explore_labels(path): 
    labels_list = os.listdir(path)
    for label in labels_list: 
        print("Shape: ", np.array(Image.open(path+"\\"+label)).shape)
        break

cycle_gan_path = "data\\preprocessing_outputs\\unified_augmented_data\\conditionnal_gan"
transformed = "data\\preprocessing_outputs\\transformed_images_labels"

synthetic_labels = "src\\data_augmentation\\gans\\base_gan\\generated_samples"
synthetic_labels_checkpoint = "src\\data_augmentation\\gans\\base_gan\\samples"

explore_images_labels(cycle_gan_path)
explore_images_labels(transformed)

#explore_labels(synthetic_labels)
#explore_labels(synthetic_labels_checkpoint)
