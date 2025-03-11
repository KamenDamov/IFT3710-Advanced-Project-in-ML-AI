import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from cycle_gan.models.pix2pix_model import Pix2PixModel
from cycle_gan.options.test_options import TestOptions
from cycle_gan.options.base_options import BaseOptions
import argparse

input_dir = "data\preprocessing_outputs\\transformed_images_labels\images"   # Raw cell images
mask_dir = "data\preprocessing_outputs\\transformed_images_labels\labels"     # Corresponding segmentation masks
output_dir = "data\Training-unlabeled\Training-unlabeled\labels" # Output paired images

def check_dir_exists(directory): 
    return os.path.exists(directory)

print(check_dir_exists(input_dir))
print(check_dir_exists(mask_dir))

os.makedirs(output_dir, exist_ok=True)

# Load Pix2Pix Model
def load_pix2pix_model(model_name="cell_segmentation_pix2pix", gpu_id=0):
    opt = BaseOptions()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = opt.initialize(parser)
    opt.name = model_name
    opt.preprocess = ""
    opt.model = "pix2pix"
    opt.direction = "AtoB"
    opt.checkpoints_dir  = "./checkpoint"
    opt.input_nc = 3
    opt.output_nc = 1
    opt.ngf = 64
    opt.ndf = 64
    opt.netG = "resnet_9blocks"
    opt.netD = "basic"
    opt.norm = "instance"
    opt.init_type = "normal"
    opt.no_dropout = ""
    opt.init_gain = 0.2
    opt.epoch = "latest"
    opt.load_iter = 0
    opt.n_layers_D = 3
    opt.gan_mode = "lsgan"
    opt.lr = 0.0002
    opt.beta1 = 0.5
    opt.lr_policy = "linear"
    opt.epoch_count = 1
    opt.n_epochs = 100
    opt.n_epochs_decay = 100
    opt.lr_decay_iters = 50
    opt.continue_train = False
    opt.verbose = True
    opt.gpu_ids = [gpu_id] if torch.cuda.is_available() else [-1]
    model = Pix2PixModel(opt)
    model.setup(opt)
    model.eval()  # Set model to evaluation mode
    return model

def generate_pseudo_masks(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue
        
        image_path = os.path.join(input_dir, filename)
        image = torch.tensor(np.array(Image.open(image_path).convert("RGB")).astype(np.float32))
        image = image.permute(2, 0, 1)
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

unlabeled_images_path = "data\preprocessing_outputs\\transformed_images_labels\images"  # Unlabeled images
pseudo_mask_output_path = "./dataset_pix2pix/test/generated_masks/"  # Where to save masks

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
