import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cycle_gan.models.pix2pix_model import Pix2PixModel
from cycle_gan.options.base_options import BaseOptions
from cycle_gan.data.produce_dataset import Pix2PixDataset
import argparse
#from models import create_model
from cycle_gan.util.visualizer import Visualizer
import time



def check_dir_exists(directory): 
    return os.path.exists(directory)

#print(check_dir_exists(input_dir))
#print(check_dir_exists(mask_dir))

#os.makedirs(output_dir, exist_ok=True)

class Options: 
    def get_opt(self, gpu_id=0): 
        opt = BaseOptions()
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = opt.initialize(parser)
        opt.name = "cell_segmentation_pix2pix"
        opt.preprocess = ""
        opt.model = "pix2pix"
        opt.direction = "BtoA"
        opt.checkpoints_dir  = "./checkpoint"
        opt.input_nc = 1
        opt.output_nc = 3
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
        opt.n_epochs = 25
        opt.n_epochs_decay = 25
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
        opt.batch_size = 8
        opt.display_freq = 400
        opt.update_html_freq = 1000
        opt.save_latest_freq = 5000
        opt.save_epoch_freq = 1
        opt.save_by_iter = True
        opt.gpu_ids = [gpu_id] if torch.cuda.is_available() else [-1]
        return opt

class Training:
        
    # Load Pix2Pix Model
    def load_pix2pix_model(self, opt):
        model = Pix2PixModel(opt)
        model.setup(opt)
        model.eval()  # Set model to evaluation mode
        return model

    def train_model(self, model, dataset, opt): 
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters = 0                # the total number of training iterations

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs
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

                # Update the loss plots
                if total_iters % opt.print_freq == 0:    # print training losses and update the plots
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    #torch.save(model.netG.state_dict(), f"{save_suffix}_generator.pth")

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                torch.save(model.netG.state_dict(), f"latest_generator.pth")

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        # Finalize the plots
        #plt.ioff()  # Turn off interactive mode
        #plt.show()


