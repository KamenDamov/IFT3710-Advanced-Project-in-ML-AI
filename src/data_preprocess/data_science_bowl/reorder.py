import os
import numpy as np
from skimage import io
import shutil

def merge_masks_into_single_label(
    input_folder, 
    output_folder, 
    image_subdir='images', 
    mask_subdir='masks'
):
    """
    Merge multiple mask images of the same base name into one label image.
    
    Parameters
    ----------
    input_folder : str
        Path to the main folder containing `images/` and `masks/`.
    output_folder : str
        Path to the output folder where merged images and labels will be saved.
    image_subdir : str, optional
        Subdirectory name for original images in `input_folder`.
    mask_subdir : str, optional
        Subdirectory name for mask images in `input_folder`.
    """
    # Create output subdirectories
    out_images_dir = os.path.join(output_folder, image_subdir)
    out_labels_dir = os.path.join(output_folder, 'labels')
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    images_dir = os.path.join(input_folder, image_subdir)
    masks_dir = os.path.join(input_folder, mask_subdir)

    # List all images in the images folder
    image_filenames = sorted(os.listdir(images_dir))
    
    for img_file in image_filenames:
        if img_file.startswith('.') or img_file.lower().endswith(('txt', 'csv')):
            # Skip hidden/system files, non-image text files, etc.
            continue

        # Base name without extension, e.g. "1" from "1.png"
        base_name = os.path.splitext(img_file)[0]
        
        # Read the original image (simply copy it later)
        image_path = os.path.join(images_dir, img_file)
        img = io.imread(image_path)  # Not actually used for merging, just read to ensure validity
        
        # Find all mask files that start with the same base name
        # e.g., "1_1.png", "1_2.png", "1_3.png", ...
        # A simple pattern is to check `startswith(base_name + "_")` or exactly `base_name`.
        mask_files = [
            f for f in os.listdir(masks_dir)
            if f.startswith(base_name + "_") or f == base_name + ".png"
        ]
        
        if not mask_files:
            # No masks found for this image => skip or create empty label?
            print(f"No mask files found for {img_file}. Skipping.")
            continue
        
        # Sort mask_files to ensure consistent labeling order
        mask_files.sort()
        
        # Read the first mask to get shape, assume all masks have same size
        first_mask_path = os.path.join(masks_dir, mask_files[0])
        first_mask = io.imread(first_mask_path)
        
        # Create an empty label image
        # Use something like np.uint16 or np.int32 if expecting many objects
        combined_label = np.zeros(first_mask.shape[:2], dtype=np.uint16)
        
        current_label_id = 1
        for mf in mask_files:
            mask_path = os.path.join(masks_dir, mf)
            mask_img = io.imread(mask_path)
            
            # If mask_img has multiple channels or alpha, use mask_img[:,:,0] or a suitable channel
            # Otherwise, assume single channel
            if mask_img.ndim == 3:
                mask_img = mask_img[:, :, 0]
                
            # Convert to boolean
            object_mask = mask_img > 0
            
            # Assign this object_mask a unique ID in the combined label
            combined_label[object_mask] = current_label_id
            current_label_id += 1
        
        # Save the combined label
        out_label_path = os.path.join(out_labels_dir, f"{base_name}.png")
        io.imsave(out_label_path, combined_label.astype(np.uint16), check_contrast=False)
        
        # Also copy the original image to output_folder/images
        out_image_path = os.path.join(out_images_dir, img_file)
        shutil.copy2(image_path, out_image_path)
        
        print(f"Merged {len(mask_files)} masks for {base_name} -> {out_label_path}")

def rename_labels_to_match_images(labels_folder):
        """
        Rename all label files in the labels folder to match the corresponding image names.
        This function assumes that the label files have "label" in their names and replaces it with "image".
        
        Parameters
        ----------
        labels_folder : str
            Path to the folder containing label files.
        """
        for label_file in os.listdir(labels_folder):
            if "label" in label_file:
                new_name = label_file.replace("label", "image")
                old_path = os.path.join(labels_folder, label_file)
                new_path = os.path.join(labels_folder, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed {label_file} -> {new_name}")

# Example usage
if __name__ == "__main__":
    input_folder = "path/to/folder"       # e.g. "folder/" that has images/ + masks/
    output_folder = "./data/data_science_bowl/labels"      # e.g. "output/"
    

    # Example usage
    rename_labels_to_match_images(output_folder)
    #merge_masks_into_single_label(
    #    input_folder=input_folder, 
    #    output_folder=output_folder, 
    #    image_subdir='images',  # name of images folder
    #    mask_subdir='masks'     # name of masks folder
    #)
