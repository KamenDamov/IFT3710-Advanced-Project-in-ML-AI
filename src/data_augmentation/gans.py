import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from models.cycle_gan.models.pix2pix_model import Pix2PixModel
from models.cycle_gan.options.test_options import TestOptions

input_dir = "data\Training-labeled\Training-labeled\images"   # Raw cell images
mask_dir = "data\Training-labeled\Training-labeled\labels"     # Corresponding segmentation masks
output_dir = "data\Training-unlabeled\Training-unlabeled\labels" # Output paired images

os.makedirs(output_dir, exist_ok=True)

# Load Pix2Pix Model
def load_pix2pix_model(model_name="cell_segmentation_pix2pix", gpu_id=0):
    opt = TestOptions().parse()
    opt.name = model_name
    opt.model = "pix2pix"
    opt.direction = "AtoB"
    opt.gpu_ids = [gpu_id] if torch.cuda.is_available() else [-1]
    model = Pix2PixModel(opt)
    model.setup(opt)
    model.eval()  # Set model to evaluation mode
    return model

def preprocess_image(image_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1] range
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0) 

def generate_pseudo_masks(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue
        
        image_path = os.path.join(input_dir, filename)
        image = preprocess_image(image_path)
        
        with torch.no_grad():
            generated = model.netG(image.to(model.device))  # Run Pix2Pix generator
        
        # Convert tensor to image
        generated = generated.squeeze().cpu().detach().numpy()
        generated = (generated + 1) * 127.5  # Convert from [-1,1] to [0,255]
        generated = np.clip(generated, 0, 255).astype(np.uint8)

        # Save pseudo-mask
        mask_output_path = os.path.join(output_dir, filename)
        cv2.imwrite(mask_output_path, generated)

        print(f"Generated pseudo-mask saved: {mask_output_path}")

unlabeled_images_path = "dataset_pix2pix/test/images/"  # Unlabeled images
pseudo_mask_output_path = "dataset_pix2pix/test/generated_masks/"  # Where to save masks
""
# Load trained Pix2Pix model
pix2pix_model = load_pix2pix_model()

# Generate pseudo-masks
generate_pseudo_masks(pix2pix_model, unlabeled_images_path, pseudo_mask_output_path)

final_images_path = "dataset/train/images/"
final_masks_path = "dataset/train/masks/"

# Move pseudo-labeled data into training dataset
for filename in os.listdir(unlabeled_images_path):
    shutil.copy(os.path.join(unlabeled_images_path, filename), final_images_path)
    shutil.copy(os.path.join(pseudo_mask_output_path, filename), final_masks_path)

print("Synthetic pseudo-masks added to training dataset!")
