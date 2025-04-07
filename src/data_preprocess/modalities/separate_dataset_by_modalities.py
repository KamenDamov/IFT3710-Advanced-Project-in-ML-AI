import os, pickle, shutil

# Load from the .pkl file
with open("modalities.pkl", "rb") as f:  # "rb" means read binary
    loaded_data = pickle.load(f)

repo_root = "C:\\Users\\guill\\OneDrive - Universite de Montreal\\HIV 2025\\Projets en apprentissage automatique\\IFT3710-Advanced-Project-in-ML-AI\\data"

# Define the original images folder
original_source = os.path.join(repo_root, "raw\\Train_Labeled")
original_images = os.path.join(original_source, "images")
original_labels = os.path.join(original_source, "labels")

# Define the destination root folder
destination_root = os.path.join(repo_root, "processed\\modalities")

# Ensure the destination root exists
os.makedirs(destination_root, exist_ok=True)

# Iterate over the dictionary
for folder_name, image_ids in loaded_data.items():
    # Create the target folder
    folder_path = os.path.join(destination_root, f'modality_{folder_name}')
    os.makedirs(folder_path, exist_ok=True)
    destination_images = os.path.join(folder_path, 'images')
    destination_labels = os.path.join(folder_path, 'labels')
    os.makedirs(destination_images, exist_ok=True)
    os.makedirs(destination_labels, exist_ok=True)

    # Copy matching images
    for image_id in image_ids:
        image_name = f"cell_{image_id:05d}.png"  # Ensure zero-padding if needed
        label_name = f"cell_{image_id:05d}_label.tiff"
        source_image_path = os.path.join(original_images, image_name)
        source_label_path = os.path.join(original_labels, label_name)
        destination_images_path = os.path.join(destination_images, image_name)
        destination_labels_path = os.path.join(destination_labels, label_name)

        # Copy if the file exists
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, destination_images_path)
            shutil.copy(source_label_path, destination_labels_path)
        else:
            print(f"Warning: {image_name} not found in {original_images}")

print("Images have been sorted into folders.")
