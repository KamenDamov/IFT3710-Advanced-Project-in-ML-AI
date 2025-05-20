import numpy as np
from PIL import Image as Image
import os

# Function to tranform labels from 256, 256, 3 => 256, 256, 1

def transform_labels(path):
    """
    Load an image from a given path.
    """
    model_type = os.listdir(path)
    for model in model_type:
        print("Model: ", model)
        label_path = path + "\\" + model + "\\labels"
        labels_list = os.listdir(label_path)
        for label in labels_list:
            label_tensor = np.array(Image.open(label_path + "\\" + label))
            print("Shape: ", label_tensor.shape)
            if len(label_tensor.shape)==3 and label_tensor.shape[2] == 3:
                label_tensor = label_tensor[:, :, 0]
                print(label_tensor.shape)
                print(label_path + "\\" + label)
                Image.fromarray(label_tensor).save(label_path + "\\" + label)
path = "data\\preprocessing_outputs\\unified_augmented_data"
transform_labels(path)