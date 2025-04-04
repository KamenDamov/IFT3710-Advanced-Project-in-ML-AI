import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools
import datetime
import argparse
import random

class CellCycleGANDataset(Dataset):
    """
    Dataset for CycleGAN training
    Loads unpaired masks and cell images
    """
    def __init__(self, mask_dir, cell_dir, transform=None):
        self.mask_dir = mask_dir
        self.cell_dir = cell_dir
        self.transform = transform
        
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.cell_files = sorted([f for f in os.listdir(cell_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        # Ensure we have data in both domains
        assert len(self.mask_files) > 0, f"No mask files found in {mask_dir}"
        assert len(self.cell_files) > 0, f"No cell files found in {cell_dir}"
    
    def __len__(self):
        # Return the length of the smaller dataset to ensure balanced training
        return min(len(self.mask_files), len(self.cell_files))
    
    def __getitem__(self, idx):
        # For masks, we'll access sequentially
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx % len(self.mask_files)])
        
        # For cells, we'll access randomly to create unpaired data
        cell_idx = random.randint(0, len(self.cell_files) - 1)
        cell_path = os.path.join(self.cell_dir, self.cell_files[cell_idx])
        
        # Load mask (domain A)
        mask = Image.open(mask_path)
        if mask.mode == 'RGB':
            mask = mask.convert('L')  # Convert to grayscale
        
        # Load cell image (domain B)
        cell = Image.open(cell_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            mask = self.transform(mask)
            cell = self.transform(cell)
            
        return {'A': mask, 'B': cell, 'A_path': mask_path, 'B_path': cell_path}

# Building blocks for the CycleGAN architecture

class ResNetBlock(nn.Module):
    """Define a Resnet block"""
    
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding {padding_type} is not implemented')
            
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding {padding_type} is not implemented')
            
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
            
        return nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = x + self.conv_block(x)  # Add skip connection
        return out

class ResNetGenerator(nn.Module):
    """Generator for CycleGAN - Transforms between domains"""
    
    def __init__(self, input_nc=1, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d, 
                 use_dropout=False, padding_type='reflect'):
        super(ResNetGenerator, self).__init__()
        
        # Set bias based on normalization layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
            
        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, 
                                 use_dropout=use_dropout, use_bias=use_bias)]
            
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                  padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
            
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Standard forward pass"""
        return self.model(input)

class PatchDiscriminator(nn.Module):
    """Discriminator for CycleGAN - PatchGAN architecture"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(PatchDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        kw = 4  # kernel width
        padw = 1  # padding width
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Final output layer
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, input):
        return self.model(input)

# Loss functions for CycleGAN
class GANLoss(nn.Module):
    """Define GAN loss"""
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
        
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
        
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

# Image buffer for replay
class ImagePool:
    """A buffer that stores generated images from previous iterations"""
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            
    def query(self, images):
        if self.pool_size == 0:  # If pool size is 0, return input images without buffering
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # Use input image with 50% probability
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

class CycleGANModel:
    """CycleGAN Model - Ties together all components"""
    
    def __init__(self, input_nc_A=1, input_nc_B=3, ngf=64, ndf=64, n_blocks=9, device='cuda'):
        self.device = device
        
        # Define networks
        # Generator A to B (mask to cell)
        self.netG_A2B = ResNetGenerator(input_nc=input_nc_A, output_nc=input_nc_B, ngf=ngf, n_blocks=n_blocks).to(device)
        
        # Generator B to A (cell to mask)
        self.netG_B2A = ResNetGenerator(input_nc=input_nc_B, output_nc=input_nc_A, ngf=ngf, n_blocks=n_blocks).to(device)
        
        # Discriminator A (for masks)
        self.netD_A = PatchDiscriminator(input_nc=input_nc_A, ndf=ndf).to(device)
        
        # Discriminator B (for cells)
        self.netD_B = PatchDiscriminator(input_nc=input_nc_B, ndf=ndf).to(device)
        
        # Image pools for historical generated images
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)
        
        # Loss functions
        self.criterionGAN = GANLoss().to(device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        
        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A2B.parameters()) + list(self.netG_B2A.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        
        self.lambda_A = 10.0  # Weight for cycle loss (A -> B -> A)
        self.lambda_B = 10.0  # Weight for cycle loss (B -> A -> B)
        self.lambda_idt = 0.5  # Weight for identity mapping loss
        
    def set_input(self, input_data):
        self.real_A = input_data['A'].to(self.device)
        self.real_B = input_data['B'].to(self.device)
        
    def forward(self):
        """Forward pass through CycleGAN"""
        # Forward pass - generate fake images
        self.fake_B = self.netG_A2B(self.real_A)  # G_A(A): 1-channel -> 3-channels
        
        # For cycle consistency
        self.rec_A = self.netG_B2A(self.fake_B)   # G_B(G_A(A)): 3-channels -> 1-channel
        
        self.fake_A = self.netG_B2A(self.real_B)  # G_B(B): 3-channels -> 1-channel
        
        # For cycle consistency
        if self.fake_A.shape[1] == 1 and getattr(self.netG_A2B.model[1], 'in_channels', 0) == 1:
            self.rec_B = self.netG_A2B(self.fake_A)  # G_A(G_B(B)): 1-channel -> 3-channels
        else:
            # Handle any other channel mismatch
            print(f"Warning: Channel mismatch - fake_A: {self.fake_A.shape[1]}, G_A2B expects: {getattr(self.netG_A2B.model[1], 'in_channels', 'unknown')}")
            self.rec_B = self.fake_A  # Fallback
        
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)
        
    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)
        
    def backward_G(self):
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        # Identity loss
        #if self.lambda_idt > 0:
        #    # For G_A2B: Requires grayscale input, but real_B is RGB
        #    # Convert real_B to grayscale for identity loss
        #    real_B_gray = torch.mean(self.real_B, dim=1, keepdim=True)  # Convert RGB to grayscale
        #    self.idt_A = self.netG_A2B(real_B_gray)  # Now passes 1-channel input to generator
        #    self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt
        #    
        #    # For G_B2A: Already expects RGB input, and outputs grayscale
        #    self.idt_B = self.netG_B2A(self.real_A)
        #    self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt
        #else:
        #    self.loss_idt_A = 0
        #    self.loss_idt_B = 0
            
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), True)
        
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_A(self.fake_A), True)
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
        
        # Combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
        
    def optimize_parameters(self):
        # Forward
        self.forward()
        
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # No need to calculate D gradients
        self.optimizer_G.zero_grad()  # Set G's gradients to zero
        self.backward_G()  # Calculate gradients for G
        self.optimizer_G.step()  # Update G's weights
        
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # Set D's gradients to zero
        self.backward_D_A()  # Calculate gradients for D_A
        self.backward_D_B()  # Calculate gradients for D_B
        self.optimizer_D.step()  # Update D's weights
        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def get_current_errors(self):
        return {
            'G_A': self.loss_G_A.item(),
            'G_B': self.loss_G_B.item(),
            'D_A': self.loss_D_A.item(),
            'D_B': self.loss_D_B.item(),
            'cycle_A': self.loss_cycle_A.item(),
            'cycle_B': self.loss_cycle_B.item(),
            'idt_A': self.loss_idt_A if isinstance(self.loss_idt_A, int) else self.loss_idt_A.item(),
            'idt_B': self.loss_idt_B if isinstance(self.loss_idt_B, int) else self.loss_idt_B.item()
        }
    
    def get_current_visuals(self):
        return {
            'real_A': self.real_A,
            'fake_B': self.fake_B,
            'rec_A': self.rec_A,
            'real_B': self.real_B,
            'fake_A': self.fake_A,
            'rec_B': self.rec_B
        }
    
    def save_networks(self, epoch, save_dir):
        """Save models to the disk with proper naming convention"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, net in zip(['G_A2B', 'G_B2A', 'D_A', 'D_B'], 
                            [self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B]):
            save_path = os.path.join(save_dir, f"{name}_epoch_{epoch}.pth")
            torch.save(net.state_dict(), save_path)
            print(f"Saved model: {save_path}")
    
    def load_networks(self, epoch, load_dir):
        """Load models from the disk"""
        for name, net in zip(['G_A2B', 'G_B2A', 'D_A', 'D_B'], 
                             [self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B]):
            load_path = os.path.join(load_dir, f"{name}_epoch_{epoch}.pth")
            net.load_state_dict(torch.load(load_path, map_location=self.device))

def train_cyclegan(model, dataloader, num_epochs=200, num_sub_epochs=3, display_freq=100, 
                   save_dir='checkpoints', sample_dir='samples'):
    """Train the CycleGAN model"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    device = model.device
    history = {
        'G_A': [], 'G_B': [], 'D_A': [], 'D_B': [], 
        'cycle_A': [], 'cycle_B': [], 'idt_A': [], 'idt_B': []
    }
    
    print(f"Starting CycleGAN training for {num_epochs} epochs")
    
    total_iters = 0
    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        epoch_iter = 0
        
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        for i, data in enumerate(dataloader):
            epoch_iter += 1

            # Sub-epoch training
            for sub_epoch in range(num_sub_epochs):
                total_iters += 1
                
                # Set input data
                model.set_input(data)
                
                # Update network parameters
                model.optimize_parameters()
                
                # Display training progress
                if total_iters % display_freq == 0:
                    errors = model.get_current_errors()
                    print(f"Epoch {epoch+1}/{num_epochs}, Iter {epoch_iter}/{len(dataloader)}")
                    for k, v in errors.items():
                        print(f"{k}: {v:.3f}", end=' ')
                    print()
                    
                    # Save sample images
                    visuals = model.get_current_visuals()
                    save_image_grid(visuals, epoch, i, sample_dir)
                
                # Update history
                for k, v in model.get_current_errors().items():
                    if k in history:
                        history[k].append(v)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({k: f"{v:.3f}" for k, v in model.get_current_errors().items()})

        pbar.close()
        
        # Print epoch summary
        time_elapsed = datetime.datetime.now() - epoch_start_time
        print(f'End of epoch {epoch+1}/{num_epochs} \t Time Taken: {time_elapsed}')
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            model.save_networks(epoch + 1, save_dir)
            print(f"Saved models at epoch {epoch+1}")

        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Get sample batch from dataloader
            sample_batch = next(iter(dataloader))
            model.set_input(sample_batch)
            model.forward()
            visuals = model.get_current_visuals()
            save_image_grid(visuals, epoch, 0, sample_dir)
            print(f"Saved samples for epoch {epoch+1}")
            
        # Update learning rates if using a scheduler
        # model.update_learning_rate()
        
    # Save final model
    model.save_networks('final', save_dir)
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    return model, history

def main():
    """Main function"""
    args = get_fixed_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # For single channel (mask)
    ])
    
    # Create dataset and dataloader
    dataset = CellCycleGANDataset(
        mask_dir=args.mask_dir,
        cell_dir=args.cell_dir,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize CycleGAN model
    model = CycleGANModel(
        input_nc_A=args.input_nc_A,  # Usually 1 for masks
        input_nc_B=args.input_nc_B,  # Usually 3 for cell images
        ngf=64,
        ndf=64,
        n_blocks=args.n_blocks,
        device=device
    )
    
    # Set cycle loss weights
    model.lambda_A = args.lambda_A
    model.lambda_B = args.lambda_B
    model.lambda_idt = args.lambda_idt
    
    # Update optimizers with custom learning rate
    model.optimizer_G = torch.optim.Adam(
        list(model.netG_A2B.parameters()) + list(model.netG_B2A.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2)
    )
    model.optimizer_D = torch.optim.Adam(
        list(model.netD_A.parameters()) + list(model.netD_B.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2)
    )
    
    if args.mode == 'train':
        print(f"Starting training with {len(dataset)} samples")
        print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
        
        # Train model
        model, history = train_cyclegan(
            model=model,
            dataloader=dataloader,
            num_epochs=args.epochs,
            save_dir=args.checkpoint_dir,
            sample_dir=args.sample_dir
        )
        
        print("Training completed")
        generate_cell_images(
            model=model,
            mask_dir=args.syn_mask_dir,  # Use the same mask directory
            output_dir=args.output_dir,
            device=device
        )
        
    elif args.mode == 'generate':
        if args.load_epoch is None:
            print("Error: Must specify --load_epoch for generation mode")
            return
        
        # Load trained model
        print(f"Loading model from epoch {args.load_epoch}")
        model.load_networks(args.load_epoch, args.checkpoint_dir)
        
        # Generate images
        print(f"Generating cell images from masks in {args.mask_dir}")
        generate_cell_images(
            model=model,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size
        )
        
        print("Generation completed")

def save_image_grid(visuals, epoch, iter_idx, save_dir):
    """Save a grid of images for visual comparison"""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    def tensor_to_image(tensor, batch_idx=0):
        # Convert tensor to numpy array, handling batches correctly
        # Pick a single sample from the batch
        tensor = tensor[batch_idx:batch_idx+1]  # Keep batch dimension but with size 1
        
        if tensor.shape[1] == 1:  # If grayscale
            image = tensor.detach().cpu().squeeze().numpy()  # Now squeeze works as expected
            return (image + 1) / 2.0
        else:  # If RGB
            image = tensor.detach().cpu().squeeze().numpy()  # Remove batch dim
            image = np.transpose(image, (1, 2, 0))  # Correct ordering for matplotlib
            return (image + 1) / 2.0
    
    # Use the first image in the batch for visualization
    batch_idx = 0
    
    # Generate grid
    axs[0, 0].imshow(tensor_to_image(visuals['real_A'], batch_idx), 
                     cmap='gray' if visuals['real_A'].shape[1] == 1 else None)
    axs[0, 0].set_title('Real Mask (A)')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(tensor_to_image(visuals['fake_B'], batch_idx))
    axs[0, 1].set_title('Generated Cell (B)')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(tensor_to_image(visuals['rec_A'], batch_idx), cmap='gray' if visuals['real_A'].shape[1] == 1 else None)
    axs[0, 2].set_title('Reconstructed Mask (A)')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(tensor_to_image(visuals['real_B'], batch_idx))
    axs[1, 0].set_title('Real Cell (B)')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(tensor_to_image(visuals['fake_A'], batch_idx), cmap='gray' if visuals['real_A'].shape[1] == 1 else None)
    axs[1, 1].set_title('Generated Mask (A)')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(tensor_to_image(visuals['rec_B'], batch_idx))
    axs[1, 2].set_title('Reconstructed Cell (B)')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}_iter_{iter_idx}.png'))
    plt.close(fig)

def plot_training_history(history, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(15, 10))
    
    # Plot generator losses
    plt.subplot(2, 2, 1)
    plt.plot(history['G_A'], label='G_A')
    plt.plot(history['G_B'], label='G_B')
    plt.title('Generator Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot discriminator losses
    plt.subplot(2, 2, 2)
    plt.plot(history['D_A'], label='D_A')
    plt.plot(history['D_B'], label='D_B')
    plt.title('Discriminator Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot cycle consistency losses
    plt.subplot(2, 2, 3)
    plt.plot(history['cycle_A'], label='cycle_A')
    plt.plot(history['cycle_B'], label='cycle_B')
    plt.title('Cycle Consistency Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot identity losses
    plt.subplot(2, 2, 4)
    plt.plot(history['idt_A'], label='idt_A')
    plt.plot(history['idt_B'], label='idt_B')
    plt.title('Identity Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_fixed_args():
    args = argparse.Namespace()
    args.syn_mask_dir = 'C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\data\\generated_samples'
    args.mask_dir = 'C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\normalized_data\\labels'
    args.cell_dir = 'C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\normalized_data\\images'
    args.output_dir = 'output'
    args.sample_dir = 'samples'
    args.checkpoint_dir = 'checkpoints'
    args.batch_size = 2
    args.epochs = 1
    args.lr = 0.0002
    args.beta1 = 0.5
    args.beta2 = 0.999
    args.lambda_A = 10.0
    args.lambda_B = 10.0
    args.lambda_idt = 0.5
    args.n_blocks = 1
    args.input_nc_A = 1
    args.input_nc_B = 3
    args.mode = 'train'  # or 'generate'
    args.load_epoch = None  # Set to an epoch number if in generate mode
    
    return args


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a CycleGAN model for cell image generation')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask images (domain A)')
    parser.add_argument('--cell_dir', type=str, required=True, help='Directory containing cell images (domain B)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--sample_dir', type=str, required=True, help='Directory to save sample images during training')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--syn_mask_dir', type=str, required=True, help='Directory of the synthetic masks')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='Weight for cycle loss (A->B->A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='Weight for cycle loss (B->A->B)')
    parser.add_argument('--lambda_idt', type=float, default=0.5, help='Weight for identity loss')
    parser.add_argument('--n_blocks', type=int, default=9, help='Number of ResNet blocks in generators')
    parser.add_argument('--input_nc_A', type=int, default=1, help='Number of channels in input masks')
    parser.add_argument('--input_nc_B', type=int, default=3, help='Number of channels in input cells')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'], help='Train model or generate images')
    parser.add_argument('--load_epoch', type=int, default=None, help='Epoch to load for inference')
    
    return parser.parse_args()

def generate_cell_images(model, mask_dir, output_dir, device, batch_size=1):
    """Generate cell images from masks using the trained CycleGAN model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.netG_A2B.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Get all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    print(f"Found {len(mask_files)} mask files to process")
    
    with torch.no_grad():
        for mask_file in tqdm(mask_files, desc="Generating Cell Images"):
            # Extract base filename without extension
            base_name = os.path.splitext(mask_file)[0]
            
            # Load and preprocess mask
            mask_path = os.path.join(mask_dir, mask_file)
            mask = Image.open(mask_path)
            if mask.mode == 'RGB':
                mask = mask.convert('L')
            
            mask_tensor = transform(mask).unsqueeze(0).to(device)
            
            # Generate cell image
            fake_cell = model.netG_A2B(mask_tensor)
            
            # Convert to image
            fake_cell = fake_cell.cpu().squeeze().numpy()
            fake_cell = np.transpose(fake_cell, (1, 2, 0))
            fake_cell = (fake_cell + 1) / 2.0 * 255.0
            fake_cell = fake_cell.astype(np.uint8)
            
            # Save the generated image with consistent naming
            output_file = os.path.join(output_dir, f"{base_name}_generated.png")
            Image.fromarray(fake_cell).save(output_file)
            
            # Create comparison visualization
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(np.array(mask), cmap='gray')
            axs[0].set_title('Input Mask')
            axs[0].axis('off')
            
            axs[1].imshow(fake_cell)
            axs[1].set_title('Generated Cell Image')
            axs[1].axis('off')
            
            plt.tight_layout()
            viz_file = os.path.join(output_dir, f"{base_name}_comparison.png")
            plt.savefig(viz_file)
            plt.close(fig)

if __name__ == "__main__":
    main()
