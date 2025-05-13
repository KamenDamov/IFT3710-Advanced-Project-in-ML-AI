import os
import shutil
import platform

def unify(base_data_path, augmented_data_path, output_path, labels_path, augment_type): 
    os.makedirs(output_path + augment_type, exist_ok=True)
    os.makedirs(output_path + augment_type + "/images", exist_ok=True)
    os.makedirs(output_path + augment_type + "/labels", exist_ok=True)
    
    # Copy augmented images and labels
    for file in os.listdir(augmented_data_path):
        if "_generated.png" in file:
            label_file = file.replace("_generated.png", ".png")
            src_img = os.path.join(augmented_data_path, file)
            dst_img = os.path.join(output_path + augment_type, "images", label_file)  # Use label_file name for image
            
            src_label = os.path.join(labels_path, label_file)
            dst_label = os.path.join(output_path + augment_type, "labels", label_file)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"Warning: Image file {src_img} not found")
                
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: Label file {src_label} not found")
    
    # Copy base dataset images and labels using shutil for better reliability
    for file in os.listdir(os.path.join(base_data_path, "images")):
        src_img = os.path.join(base_data_path, "images", file)
        dst_img = os.path.join(output_path + augment_type, "images", file)
        if not os.path.exists(dst_img):  
            shutil.copy2(src_img, dst_img)
            
    for file in os.listdir(os.path.join(base_data_path, "labels")):
        src_label = os.path.join(base_data_path, "labels", file)
        dst_label = os.path.join(output_path + augment_type, "labels", file)
        if not os.path.exists(dst_label):
            shutil.copy2(src_label, dst_label)

labels_path = "src/data_augmentation/gans/base_gan/generated_samples"
base_training_data_path = "data/preprocessing_outputs/transformed_images_labels"
cycle_gan_augmented_training_data_path = "output_cycle_gan"
conditionnal_gan_augmented_training_data_path = "output_attention_gan"
attention_gan_augmented_training_data_path = "synthetic_samples_unet150"
output_path = "data/preprocessing_outputs/unified_augmented_data"

unify(base_training_data_path, cycle_gan_augmented_training_data_path, output_path, labels_path, "/cycle_gan")
unify(base_training_data_path, conditionnal_gan_augmented_training_data_path, output_path, labels_path, "/conditionnal_gan")
unify(base_training_data_path, attention_gan_augmented_training_data_path, output_path, labels_path, "/attention_gan")