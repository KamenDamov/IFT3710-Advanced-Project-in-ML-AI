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
import wandb
import datetime
import argparse

class CellGANDataset(Dataset):
    """
    Dataset for conditional GAN training
    Pairs synthetic masks (condition) with real cell images
    """
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform else transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load real image (target)
        real_image = Image.open(img_path).convert('RGB')
        
        # Load mask (condition)
        # Convert mask to grayscale (1 channel) if it's RGB
        mask = Image.open(mask_path)
        if mask.mode == 'RGB':
            mask = mask.convert('L')  # Convert to grayscale
        
        # Apply transforms
        if self.transform:
            real_image = self.transform(real_image)
            mask = self.mask_transform(mask)
            
        return mask, real_image


# Define a ResNet block for the ResNet generator
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

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
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        # Store the original input
        identity = x.clone()
        # Apply the convolutional block
        out = self.conv_block(x)
        # Add the skip connection (identity path)
        out = out + identity
        return out



class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    
    Adapted from the Pix2Pix implementation
    """

    def __init__(self, input_nc=1, output_nc=3, ngf=256, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator
        
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, 
                                 use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)
    
# Factory function to choose generator architecture
def get_generator(arch_type='resnet', **kwargs):
    """
    Factory function to create a generator based on architecture type
    
    Parameters:
        arch_type (str) -- 'resnet' or 'unet'
        **kwargs -- arguments to pass to the generator constructor
    """
    if arch_type.lower() == 'resnet':
        return ResnetGenerator(**kwargs)
    elif arch_type.lower() == 'small_unet':
        return UNetGenerator(
            input_nc=1,
            output_nc=3,
            ngf=32,     # Fewer base filters
            max_filters=256,
            depth=7     # Slightly less depth
        )
    elif arch_type.lower() == 'big_unet':
        return  UNetGenerator(
            input_nc=1,
            output_nc=3,
            ngf=64,
            max_filters=1024,
            depth=8
        )
    else:
        raise ValueError(f"Unknown generator architecture: {arch_type}")

# Discriminator Network
class PatchGANDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator
    This is a 70x70 PatchGAN as used in Pix2Pix
    """
    def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Construct a PatchGAN discriminator
        
        Parameters:
            input_nc (int)  -- the number of channels in input images (1 for mask + 3 for RGB image = 4)
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGANDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
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
        
        # Output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)
        
    def forward(self, mask, image):
        """Standard forward."""
        # Concatenate mask and image along channel dimension
        x = torch.cat([mask, image], dim=1)
        return self.model(x)

import torch.nn.functional as F

class UNetGenerator(nn.Module):
    """
    U-Net Generator with dynamic sizing capabilities
    """
    def __init__(self, input_nc=1, output_nc=3, ngf=64, max_filters=512, 
                 depth=8, use_dropout=True, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            input_nc (int) -- input channels (1 for mask)
            output_nc (int) -- output channels (3 for RGB)
            ngf (int) -- number of base filters (increase for larger models)
            max_filters (int) -- maximum number of filters in any layer
            depth (int) -- depth of the U-Net (8 is original, can be 6-9)
            use_dropout (bool) -- whether to use dropout
            norm_layer -- normalization layer type
        """
        super(UNetGenerator, self).__init__()
        
        # Create individual layers directly instead of dynamically constructing them
        # This ensures we have precise control over channel dimensions
        
        # Encoder path - same as your original UNet
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Add more encoder layers depending on depth parameter
        if depth > 4:
            self.enc5 = nn.Sequential(
                nn.Conv2d(ngf * 8, min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        if depth > 5:
            self.enc6 = nn.Sequential(
                nn.Conv2d(min(ngf * 8 * 2, max_filters), min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        if depth > 6:
            self.enc7 = nn.Sequential(
                nn.Conv2d(min(ngf * 8 * 2, max_filters), min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        if depth > 7:
            self.enc8 = nn.Sequential(
                nn.Conv2d(min(ngf * 8 * 2, max_filters), min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Decoder path
        # Calculate the exact input size for each decoder based on depth
        if depth > 7:
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(min(ngf * 8 * 2, max_filters), min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(min(ngf * 8 * 2, max_filters) * 2, min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        elif depth > 6:
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(min(ngf * 8 * 2, max_filters), min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        
        if depth > 6:
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(min(ngf * 8 * 2, max_filters) * 2, min(ngf * 8 * 2, max_filters), kernel_size=4, stride=2, padding=1),
                norm_layer(min(ngf * 8 * 2, max_filters)),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.ReLU(inplace=True)
            )
        
        if depth > 5:
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(min(ngf * 8 * 2, max_filters) * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
                norm_layer(ngf * 8),
                nn.ReLU(inplace=True)
            )
        
        if depth > 4:
            self.dec5 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                norm_layer(ngf * 4),
                nn.ReLU(inplace=True)
            )
        
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(inplace=True)
        )
        
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.depth = depth
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Apply encoder layers conditionally based on depth
        if self.depth > 4:
            e5 = self.enc5(e4)
        if self.depth > 5:
            e6 = self.enc6(e5)
        if self.depth > 6:
            e7 = self.enc7(e6)
        if self.depth > 7:
            e8 = self.enc8(e7)
        
        # Decoder with skip connections - also conditional based on depth
        if self.depth > 7:
            d1 = self.dec1(e8)
            d2 = self.dec2(torch.cat([d1, e7], 1))
            d3 = self.dec3(torch.cat([d2, e6], 1))
            d4 = self.dec4(torch.cat([d3, e5], 1))
            d5 = self.dec5(torch.cat([d4, e4], 1))
        elif self.depth > 6:
            d1 = self.dec1(e7)
            d3 = self.dec3(torch.cat([d1, e6], 1))
            d4 = self.dec4(torch.cat([d3, e5], 1))
            d5 = self.dec5(torch.cat([d4, e4], 1))
        elif self.depth > 5:
            d4 = self.dec4(e6)
            d5 = self.dec5(torch.cat([d4, e4], 1))
        elif self.depth > 4:
            d5 = self.dec5(e5)
        else:
            d5 = e4
            
        d6 = self.dec6(torch.cat([d5, e3], 1))
        d7 = self.dec7(torch.cat([d6, e2], 1))
        d8 = self.dec8(torch.cat([d7, e1], 1))
        
        return d8

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):  # 1 for mask + 3 for image
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # input: (in_channels) x 256 x 256
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 32 x 32
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 16 x 16
            
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 8 x 8
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            # 1 x 7 x 7
        )
    
    def forward(self, mask, image):
        # Concatenate mask and image along channel dimension
        x = torch.cat([mask, image], dim=1)
        return self.model(x)

# Loss functions
def generator_loss(fake_output, fake_image, real_image, lambda_L1=100):
    """
    Generator loss: adversarial loss + L1 loss
    """
    adversarial_loss = F.binary_cross_entropy_with_logits(
        fake_output, torch.ones_like(fake_output)
    )
    
    # L1 loss (to encourage less blurring)
    l1_loss = F.l1_loss(fake_image, real_image)
    
    # Total generator loss
    total_loss = adversarial_loss + lambda_L1 * l1_loss
    
    return total_loss, adversarial_loss, l1_loss


def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss
    """
    real_loss = F.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output)
    )
    
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output)
    )
    
    total_loss = real_loss + fake_loss
    
    return total_loss

def train_gan(generator, discriminator, train_loader, val_loader, device,
              epochs=100, lr=0.0002, beta1=0.5, beta2=0.999, lambda_L1=100,
              sample_dir='samples', checkpoint_dir='checkpoints',
              test_mask_path=None):

    
    # Training history
    history = {
        'gen_loss': [], 'disc_loss': [], 'val_gen_loss': [], 'val_disc_loss': []
    }
    
    for epoch in range(epochs):
        # Your existing training loop
        
        # After computing epoch losses, log to wandb
        
        
        # If you generate sample images during training, log those too
        if test_mask_path and epoch % 10 == 0:  # Adjust frequency as needed
            # Generate your sample image
            sample_img = generate_sample_image(generator, test_mask_path, device)
            
            # Log the image to wandb
            wandb.log({f"sample_epoch_{epoch}": wandb.Image(sample_img)})
        
        # If you save checkpoints, you can also log those to wandb
        if epoch % 50 == 0:  # Adjust frequency as needed
            # Save your checkpoint as usual
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'disc_optimizer': disc_optimizer.state_dict(),
                'epoch': epoch
            }, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
            
            # Log the checkpoint to wandb
            wandb.save(f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
    
    # Finish the wandb run when training is complete
    wandb.finish()
    
    return generator, discriminator, history

def train_gan(generator, discriminator, train_loader, val_loader, device,
              epochs=100, lr=0.0002, beta1=0.5, beta2=0.999, lambda_L1=100,
              sample_dir='samples', checkpoint_dir='checkpoints',
              test_mask_path=None):
    """
    Train the conditional GAN
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
        lambda_L1: Weight for L1 loss
        sample_dir: Directory to save sample images
        checkpoint_dir: Directory to save model checkpoints
        test_mask_path: Path to a test mask to visualize progress (optional)
    
    Returns:
        Trained generator, discriminator, and training history
    """
    # Create directories if they don't exist
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Initialize counters for discriminator skip tracking
    skipped_updates = 0
    
    # Optimizers with weight decay
    generator_optimizer = optim.Adam(
        generator.parameters(), 
        lr=lr, 
        betas=(beta1, beta2),
        weight_decay=1e-5
    )
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), 
        lr=lr, 
        betas=(beta1, beta2),
        weight_decay=1e-5
    )
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    wandb.watch(generator)
    wandb.watch(discriminator)

    # Training history
    history = {
        'generator_loss': [],
        'discriminator_loss': [],
        'adv_loss': [],
        'val_loss': [],
        'skipped_updates': []
    }
    
    # Sample images for visualization from validation set
    val_masks, val_images = next(iter(val_loader))
    fixed_masks = val_masks[:8].to(device)
    fixed_real_images = val_images[:8].to(device)
    
    # Load test mask if provided
    test_mask_tensor = None
    if test_mask_path and os.path.exists(test_mask_path):
        try:
            # Load and preprocess test mask
            test_mask = Image.open(test_mask_path)
            if test_mask.mode == 'RGB':
                test_mask = test_mask.convert('L')  # Convert to grayscale
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            
            test_mask_tensor = transform(test_mask).unsqueeze(0).to(device)
            print(f"Loaded test mask from {test_mask_path} for progress visualization")
        except Exception as e:
            print(f"Error loading test mask: {e}")
            test_mask_tensor = None
    
    # Training loop
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        # Metrics for epoch
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_adv_loss = 0
        epoch_l1_loss = 0
        epoch_skipped = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for masks, real_images in train_loader:
                masks = masks.to(device)
                real_images = real_images.to(device)
                batch_size = masks.size(0)
                
                # -----------------------
                # Train Discriminator
                # -----------------------
                discriminator_optimizer.zero_grad()
                
                # Generate fake images
                fake_images = generator(masks)
                
                # Real samples
                real_output = discriminator(masks, real_images)
                real_loss = F.binary_cross_entropy_with_logits(
                    real_output, torch.ones_like(real_output)
                )
                
                # Fake samples
                fake_output = discriminator(masks, fake_images.detach())
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_output, torch.zeros_like(fake_output)
                )
                
                # Total discriminator loss
                d_loss = real_loss + fake_loss
                
                # Skip discriminator update if it's too strong
                if d_loss.item() < 0.2:
                    discriminator_skip = True
                    epoch_skipped += 1
                else:
                    discriminator_skip = False
                    d_loss.backward()
                    discriminator_optimizer.step()
                
                # -----------------------
                # Train Generator
                # -----------------------
                generator_optimizer.zero_grad()
                
                # Generate fake images again
                fake_images = generator(masks)
                
                # Discriminator output on fake images
                fake_output = discriminator(masks, fake_images)
                
                # Generator losses
                g_loss, adv_loss, l1_loss = generator_loss(
                    fake_output, fake_images, real_images, lambda_L1
                )
                g_loss.backward()
                generator_optimizer.step()
                
                # If discriminator is weak, train generator again
                if d_loss.item() > 1.0:
                    generator_optimizer.zero_grad()
                    fake_images = generator(masks)
                    fake_output = discriminator(masks, fake_images)
                    g_loss, adv_loss, l1_loss = generator_loss(
                        fake_output, fake_images, real_images, lambda_L1
                    )
                    g_loss.backward()
                    generator_optimizer.step()
                
                # Update metrics
                epoch_g_loss += g_loss.item() * batch_size
                epoch_d_loss += d_loss.item() * batch_size
                epoch_adv_loss += adv_loss.item() * batch_size
                epoch_l1_loss += l1_loss.item() * batch_size
                
                pbar.update(1)
                pbar.set_postfix({
                    'G Loss': g_loss.item(),
                    'D Loss': d_loss.item(),
                    'Skipped': epoch_skipped
                })
        
        # Calculate epoch metrics
        epoch_g_loss /= len(train_loader.dataset)
        epoch_d_loss /= len(train_loader.dataset)
        epoch_adv_loss /= len(train_loader.dataset)
        epoch_l1_loss /= len(train_loader.dataset)
        skipped_updates += epoch_skipped
        
        # Learning rate adjustment
        if epoch in [100, 200, 300]:
            for param_group in generator_optimizer.param_groups:
                param_group['lr'] *= 0.5
            for param_group in discriminator_optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Learning rate reduced by half at epoch {epoch+1}")
        
        # Validation
        generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for masks, real_images in val_loader:
                masks = masks.to(device)
                real_images = real_images.to(device)
                batch_size = masks.size(0)
                
                # Generate fake images
                fake_images = generator(masks)
                
                # L1 loss for validation
                loss = F.l1_loss(fake_images, real_images)
                val_loss += loss.item() * batch_size
                
            val_loss /= len(val_loader.dataset)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), f'{checkpoint_dir}/best_generator.pth')
            torch.save(discriminator.state_dict(), f'{checkpoint_dir}/best_discriminator.pth')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        
        # Update history
        history['generator_loss'].append(epoch_g_loss)
        history['discriminator_loss'].append(epoch_d_loss)
        history['adv_loss'].append(epoch_adv_loss)
        history['val_loss'].append(val_loss)
        history['skipped_updates'].append(epoch_skipped)
        
        # Print epoch stats
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"G Loss: {epoch_g_loss:.4f}, "
              f"D Loss: {epoch_d_loss:.4f}, "
              f"Adv Loss: {epoch_adv_loss:.4f}, "
              f"L1 Loss: {epoch_l1_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Skipped D Updates: {epoch_skipped}")
        
        # Generate and save sample images from validation set
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generate_and_save_samples(
                generator, fixed_masks, fixed_real_images, epoch, save_dir=sample_dir
            )
        
        # Generate and save test mask result if provided
        if test_mask_tensor is not None:
            with torch.no_grad():
                # Generate image from test mask
                fake_image = generator(test_mask_tensor)
                
                # Convert to numpy for visualization
                fake_image = fake_image.cpu().squeeze().numpy()
                fake_image = np.transpose(fake_image, (1, 2, 0))
                fake_image = (fake_image + 1) / 2.0
                fake_image = np.clip(fake_image, 0, 1)
                
                # Convert test mask to numpy
                test_mask_np = test_mask_tensor.cpu().squeeze().numpy()
                test_mask_np = (test_mask_np + 1) / 2.0
                
                # Create a figure
                plt.figure(figsize=(10, 5))
                
                # Plot mask
                plt.subplot(1, 2, 1)
                plt.imshow(test_mask_np, cmap='gray')
                plt.title('Test Mask')
                plt.axis('off')
                
                # Plot generated image
                plt.subplot(1, 2, 2)
                plt.imshow(fake_image)
                plt.title(f'Generated Image (Epoch {epoch+1})')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{sample_dir}/test_progress_epoch_{epoch+1}.png')
                plt.close()
                wandb.log({f"smaple_epoch_{epoch}": wandb.Image(fake_image)})

        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict(),
                'g_loss': epoch_g_loss,
                'd_loss': epoch_d_loss,
                'val_loss': val_loss,
                'history': history
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')

            wandb.save(f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')
                        
            # Save latest model separately
            torch.save(generator.state_dict(), f'{checkpoint_dir}/latest_generator.pth')
            torch.save(discriminator.state_dict(), f'{checkpoint_dir}/latest_discriminator.pth')

        wandb.log({
            "epoch": epoch,
            "gen_loss": epoch_g_loss,
            "disc_loss": epoch_d_loss,
            "adv_loss": epoch_adv_loss,
            "val_loss": val_loss,
            "skipped_updates": skipped_updates
        })
    # Save final models
    torch.save(generator.state_dict(), f'{checkpoint_dir}/final_generator.pth')
    torch.save(discriminator.state_dict(), f'{checkpoint_dir}/final_discriminator.pth')
    wandb.finish()
    return generator, discriminator, history

def generate_and_save_samples(generator, masks, real_images, epoch, save_dir='samples'):
    """
    Generate sample images using current generator
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    generator.eval()
    with torch.no_grad():
        fake_images = generator(masks)
    
    # Convert tensors to numpy for visualization
    masks = masks.cpu().numpy()
    real_images = real_images.cpu().numpy()
    fake_images = fake_images.cpu().numpy()
    
    # Transpose tensors from [B, C, H, W] to [B, H, W, C] for plotting
    masks = np.transpose(masks, (0, 2, 3, 1))
    real_images = np.transpose(real_images, (0, 2, 3, 1))
    fake_images = np.transpose(fake_images, (0, 2, 3, 1))
    
    # Rescale images from [-1, 1] to [0, 1]
    real_images = (real_images + 1) / 2.0
    fake_images = (fake_images + 1) / 2.0
    
    # Plot images
    n_samples = masks.shape[0]
    plt.figure(figsize=(15, 5 * n_samples))
    
    for i in range(n_samples):
        # Plot mask
        plt.subplot(n_samples, 3, i * 3 + 1)
        if masks.shape[3] == 1:  # If mask is single channel
            plt.imshow(masks[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(masks[i])
        plt.title('Mask')
        plt.axis('off')
        
        # Plot real image
        plt.subplot(n_samples, 3, i * 3 + 2)
        plt.imshow(real_images[i])
        plt.title('Real Image')
        plt.axis('off')
        
        # Plot fake image
        plt.subplot(n_samples, 3, i * 3 + 3)
        plt.imshow(fake_images[i])
        plt.title('Generated Image')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/samples_epoch_{epoch+1}.png')
    plt.close()


def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(15, 5))
    
    # Plot generator and discriminator loss
    plt.subplot(1, 2, 1)
    plt.plot(history['generator_loss'], label='Generator Loss')
    plt.plot(history['discriminator_loss'], label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss')
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['val_loss'], label='Validation Loss (L1)')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.title('Validation L1 Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def inference(generator, mask_path, device):
    """
    Generate a cell image from a single mask file
    """
    # Load and preprocess mask
    mask = Image.open(mask_path)
    if mask.mode == 'RGB':
        mask = mask.convert('L')  # Convert to grayscale
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    mask_tensor = transform(mask).unsqueeze(0).to(device)
    
    # Generate image
    generator.eval()
    with torch.no_grad():
        fake_image = generator(mask_tensor)
    
    # Convert to numpy for visualization
    fake_image = fake_image.cpu().squeeze().numpy()
    fake_image = np.transpose(fake_image, (1, 2, 0))
    
    # Rescale from [-1, 1] to [0, 1]
    fake_image = (fake_image + 1) / 2.0
    
    # Clip values to valid image range
    fake_image = np.clip(fake_image, 0, 1)
    
    return fake_image


def process_mask_directory(generator, mask_dir, output_dir, device, make_comparison=True):
    """
    Process all mask files in a directory and save the generated images
    
    Args:
        generator: Trained generator model
        mask_dir: Directory containing mask files
        output_dir: Directory to save generated images
        device: Device to run inference on
        make_comparison: Whether to create comparison images with masks side by side
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    print(f"Found {len(mask_files)} mask files to process.")
    
    # Process each mask and save the generated image
    for mask_file in tqdm(mask_files, desc="Generating images"):
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Generate image
        generated_image = inference(generator, mask_path, device)
        
        # Convert to PIL image for saving
        pil_image = Image.fromarray((generated_image * 255).astype(np.uint8))
        
        # Create output filename
        output_filename = os.path.splitext(mask_file)[0] + "_generated.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the generated image
        pil_image.save(output_path)
        
        if make_comparison:
            # Create a visualization comparing input and output
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(Image.open(mask_path).convert('L'), cmap='gray')
            plt.title('Input Mask')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(generated_image)
            plt.title('Generated Cell Image')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            viz_filename = os.path.splitext(mask_file)[0] + "_comparison.png"
            viz_path = os.path.join(output_dir, viz_filename)
            plt.savefig(viz_path)
            plt.close()
    
    print(f"Successfully generated {len(mask_files)} images in {output_dir}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train a cGAN model for cell image generation')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing the masks/labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--sample_dir', type=str, required=True, help='Directory to save sample images during training')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--lambda_L1', type=float, default=150, help='Weight for L1 loss')
    parser.add_argument('--test_mask_path', type=str, default=None, help='Path to test mask for generating samples')
    return parser.parse_args()


def main():
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


def main(): 
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project="cell-gan",
        config={
            "architecture": "big_unet",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "lambda_L1": args.lambda_L1,
            "input_nc": 1,
            "output_nc": 3,
            "ngf": 256,
            "use_dropout": True,
            "n_blocks": 1
        }
    )
    
    # Directories for your data
    image_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\unified_set\\images"
    mask_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\unified_set\\labels"

    # Output directories
    sample_dir = "big_unet"
    checkpoint_dir = "checkpoints"
    output_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\dataset_pix2pix\\new_samples_big_unet"

    # Test mask for progress visualization during training
    test_mask_path = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\src\\data_augmentation\\gans\\base_gan\\generated_samples\\sample_1_epoch_86.png"

    # Initialize wandb before any training happens
    wandb.login()  # You'll need to enter your API key on first run
    wandb.init(
        project="cell-gan",  # Choose an appropriate project name
        name=f"gan-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "architecture": "big_unet",
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "beta1": beta1,
            "beta2": beta2,
            "lambda_L1": lambda_L1,
            "input_nc": 1,
            "output_nc": 3,
            "ngf": 256,
            "use_dropout": True,
            "n_blocks": 1
        }
    )

    # The rest of your code stays the same until the train_gan function

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations
    # For images: scale to [-1, 1]
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For masks: grayscale and scale to [-1, 1]
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create dataset
    dataset = CellGANDataset(
        image_dir, mask_dir, 
        transform=image_transform, 
        mask_transform=mask_transform
    )

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize ResNet generator
    generator = get_generator(
        arch_type='resnet',
        input_nc=1,          # mask channels
        output_nc=3,         # cell image
        ngf=256,             
        norm_layer=nn.InstanceNorm2d, 
        use_dropout=True,
        n_blocks=9           # Increase number of ResNet blocks for more parameters
    )

    # Initialize PatchGAN discriminator
    discriminator = PatchGANDiscriminator(
        input_nc=4,          # 1 for mask + 3 for image
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d
    )

    # Print model sizes
    print(f"Generator Architecture: Big Unet")
    print(f"Generator Parameters: {count_parameters(generator):,}")
    print(f"Discriminator Parameters: {count_parameters(discriminator):,}")

    # Create directories if they don't exist
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Train models
    # Train models
    trained_generator, trained_discriminator, history = train_gan(
        generator, discriminator, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, beta1=args.beta1, beta2=args.beta2, 
        lambda_L1=args.lambda_L1, sample_dir=args.sample_dir, 
        checkpoint_dir=args.checkpoint_dir, test_mask_path=args.test_mask_path
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Test inference on masks in a directory and save generated images
    process_mask_directory(
        trained_generator, 
        args.mask_dir,  # You can change this to a different test directory if needed
        args.output_dir, 
        device, 
        make_comparison=True
    )
    
    # Finish wandb logging
    wandb.finish()

