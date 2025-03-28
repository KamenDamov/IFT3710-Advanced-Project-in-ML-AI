import numpy as np
import os
import shutil

def get_corresponding_label(image_name, syn_labels_dir):
    """Find the label file that corresponds to an image file"""
    labels = os.listdir(syn_labels_dir)
    image_base = image_name.split(".")[0]
    
    for label in labels:
        if image_base.replace("_generated", "") in label.split(".")[0]:  
            return label
    return None
        
def create_dataset(syn_images_dir, syn_labels_dir, real_images_dir, real_labels_dir): 
    """Move and rename synthetic images and labels to the real dataset directories"""
    
    # Ensure destination directories exist
    os.makedirs(real_images_dir, exist_ok=True)
    os.makedirs(real_labels_dir, exist_ok=True)
    
    syn_images_list = [f for f in os.listdir(syn_images_dir) if f.endswith("_generated.png")]
    
    for i, image_name in enumerate(syn_images_list):
        label_name = get_corresponding_label(image_name, syn_labels_dir)
        
        if label_name is not None:
            new_name = f"{str(i)}.png"
            
            shutil.copy2(
                os.path.join(syn_images_dir, image_name), 
                os.path.join(real_images_dir, new_name)
            )
            
            shutil.copy2(
                os.path.join(syn_labels_dir, label_name), 
                os.path.join(real_labels_dir, new_name)
            )
            
            print(f"Moved file pair {i}: {image_name} and {label_name}")
        else:
            print(f"No matching label found for {image_name}")

if __name__ == "__main__":
    syn_labels = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\src\\data_augmentation\\gans\\base_gan\\generated_samples"
    syn_images = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\dataset_pix2pix\\new_samples_mod"
    real_labels = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\augmented_dataset\\labels"
    real_images = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\augmented_dataset\\images"
    
    create_dataset(syn_images, syn_labels, real_images, real_labels)