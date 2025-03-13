import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from cycle_gan.models.pix2pix_model import Pix2PixModel
from cycle_gan.options.base_options import BaseOptions
from cycle_gan.data.produce_dataset import Pix2PixDataset
import argparse
#from models import create_model
from cycle_gan.util.visualizer import Visualizer
import time

input_dir = "data\preprocessing_outputs\\transformed_images_labels\images"   # Raw cell images
mask_dir = "data\preprocessing_outputs\\transformed_images_labels\labels"     # Corresponding segmentation masks
input_tuning_dir = "data\Tuning\images"
output_dir = "data\Training-unlabeled\Training-unlabeled\labels" # Output paired images

def check_dir_exists(directory): 
    return os.path.exists(directory)

print(check_dir_exists(input_dir))
print(check_dir_exists(mask_dir))

os.makedirs(output_dir, exist_ok=True)

def get_opt(model_name="cell_segmentation_pix2pix", gpu_id=0): 
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
    opt.n_epochs = 10
    opt.n_epochs_decay = 0
    opt.lr_decay_iters = 50
    opt.continue_train = False
    opt.verbose = True

    # Visualization params
    opt.display_id = 1
    opt.display_winsize = 256
    opt.display_port = 8000
    opt.use_wandb = True
    opt.display_ncols = 4
    opt.no_html = True
    opt.display_server = "http://localhost"
    opt.wandb_project_name = "CycleGAN-and-pix2pix"
    opt.display_env = "main"
    opt.print_freq = 100
    opt.batch_size = 1
    opt.display_freq = 400
    opt.update_html_freq = 1000
    opt.save_latest_freq = 5000
    opt.save_epoch_freq = 5
    opt.save_by_iter = True
    opt.gpu_ids = [gpu_id] if torch.cuda.is_available() else [-1]
    return opt

# Load Pix2Pix Model
def load_pix2pix_model(opt):
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

def train_model(model, dataset, opt): 
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    #model = create_model(opt)      # create a model given opt.model and other options
    #model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            model.save_networks("test", "test")
            torch.save(pix2pix_model.netG.state_dict(), "latest_net_Gtest.pth")
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                #model.compute_visuals()
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                #if opt.display_id > 0:
                    #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


unlabeled_images_path = "data\preprocessing_outputs\\transformed_images_labels\images"  # Unlabeled images
pseudo_mask_output_path = "./dataset_pix2pix/test/generated_masks/"  # Where to save masks
pseudo_mask_output_tuning_path = "./dataset_pix2pix/tuning/generated_masks/"

# Load trained Pix2Pix model
opt = get_opt()
pix2pix_model = load_pix2pix_model(opt)
dataset = Pix2PixDataset(input_dir, mask_dir)
# Train pix2pix 
#train_model(pix2pix_model, dataset, opt)

pix2pix_model = Pix2PixModel(opt)
checkpoint_path = "latest_net_Gtest.pth"

# Load state dictionary
pix2pix_model.load_networks("latest")

pix2pix_model.eval()
print("Model loaded successfully!")

# Generate pseudo-masks
generate_pseudo_masks(pix2pix_model, input_tuning_dir, pseudo_mask_output_tuning_path)

final_images_path = "dataset/train/images/"
final_masks_path = "dataset/train/masks/"

print("Synthetic pseudo-masks added to training dataset!")
