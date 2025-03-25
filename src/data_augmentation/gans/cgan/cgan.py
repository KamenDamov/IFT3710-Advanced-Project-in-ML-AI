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


# Generator Network (U-Net style)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 128x128
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 64x64
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 32x32
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 16x16
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 8x8
        
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 4x4
        
        self.enc7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 2x2
        
        self.enc8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )  # 1x1
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )  # 2x2
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )  # 4x4
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )  # 8x8
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 16x16
        
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(512 * 2, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 32x32
        
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(256 * 2, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 64x64
        
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 128x128
        
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # 256x256
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], 1))
        d3 = self.dec3(torch.cat([d2, e6], 1))
        d4 = self.dec4(torch.cat([d3, e5], 1))
        d5 = self.dec5(torch.cat([d4, e4], 1))
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
              epochs=100, lr=0.0002, beta1=0.5, beta2=0.999, lambda_L1=100):
    """
    Train the conditional GAN
    """
    # Optimizers
    #generator_optimizer = optim.Adam(
    #    generator.parameters(), lr=lr, betas=(beta1, beta2)
    #)
    #discriminator_optimizer = optim.Adam(
    #    discriminator.parameters(), lr=lr, betas=(beta1, beta2)
    #)
    
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
    print(generator)
    print(f"Generator parameters: {count_parameters(generator)}")
    print(discriminator)
    print(f"Discriminator parameters: {count_parameters(discriminator)}")
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Training history
    history = {
        'generator_loss': [],
        'discriminator_loss': [],
        'val_loss': []
    }
    
    # Sample images for visualization
    val_masks, val_images = next(iter(val_loader))
    fixed_masks = val_masks[:8].to(device)
    fixed_real_images = val_images[:8].to(device)
    
    # Training loop
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        # Metrics for epoch
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_adv_loss = 0
        epoch_l1_loss = 0
        
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
                if d_loss.item() < 0.2:  # Very low discriminator loss
                    discriminator_skip = True
                else:
                    discriminator_skip = False
                    
                if not discriminator_skip:
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
                
                # Update metrics
                epoch_g_loss += g_loss.item() * batch_size
                epoch_d_loss += d_loss.item() * batch_size
                epoch_adv_loss += adv_loss.item() * batch_size
                epoch_l1_loss += l1_loss.item() * batch_size
                
                pbar.update(1)
                pbar.set_postfix({
                    'G Loss': g_loss.item(),
                    'D Loss': d_loss.item()
                })
        
        # Calculate epoch metrics
        epoch_g_loss /= len(train_loader.dataset)
        epoch_d_loss /= len(train_loader.dataset)
        epoch_adv_loss /= len(train_loader.dataset)
        epoch_l1_loss /= len(train_loader.dataset)
        if epoch == 200:
            for param_group in generator_optimizer.param_groups:
                param_group['lr'] *= 0.5
            for param_group in discriminator_optimizer.param_groups:
                param_group['lr'] *= 0.5
            print("Learning rate reduced by half")
        
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
        
        # Update history
        history['generator_loss'].append(epoch_g_loss)
        history['discriminator_loss'].append(epoch_d_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch stats
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"G Loss: {epoch_g_loss:.4f}, "
              f"D Loss: {epoch_d_loss:.4f}, "
              f"Adv Loss: {epoch_adv_loss:.4f}, "
              f"L1 Loss: {epoch_l1_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Generate and save sample images
        if (epoch + 1) % 5 == 0:
            generate_and_save_samples(
                generator, fixed_masks, fixed_real_images, epoch, save_dir='samples_mod'
            )
            torch.save(generator.state_dict(), f'cell_gan_generator_{str(epoch)}.pth')
    
    # Save final models
    torch.save(generator.state_dict(), 'cell_gan_generator.pth')
    torch.save(discriminator.state_dict(), 'cell_gan_discriminator.pth')
    
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


def main():
    # Parameters
    batch_size = 8      # This is fine for most GPUs
    epochs = 400        # Increase this since your model is still improving
    lr = 0.0001         # This lower learning rate is good
    beta1 = 0.5         # Standard for GANs
    beta2 = 0.999       # Standard value
    lambda_L1 = 150 
    
    # Directories for your data
    image_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\unified_set\\images"
    mask_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\unified_set\\labels"
    
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
    
    # Initialize models
    generator = Generator(in_channels=1, out_channels=3)
    discriminator = Discriminator(in_channels=4)  # 1 for mask + 3 for image
    
    # Train models
    trained_generator, trained_discriminator, history = train_gan(
        generator, discriminator, train_loader, val_loader, device,
        epochs=epochs, lr=lr, beta1=beta1, beta2=beta2, lambda_L1=lambda_L1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Test inference on masks in a directory and save generated images
    test_mask_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\src\\data_augmentation\\gans\\base_gan\\generated_samples"
    output_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\dataset_pix2pix\\new_samples_mod"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all mask files
    mask_files = [f for f in os.listdir(test_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    print(f"Found {len(mask_files)} mask files to process.")
    
    # Process each mask and save the generated image
    for mask_file in tqdm(mask_files, desc="Generating images"):
        mask_path = os.path.join(test_mask_dir, mask_file)
        
        # Generate image
        generated_image = inference(trained_generator, mask_path, device)
        
        # Convert to PIL image for saving
        pil_image = Image.fromarray((generated_image * 255).astype(np.uint8))
        
        # Create output filename
        output_filename = os.path.splitext(mask_file)[0] + "_generated.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the generated image
        pil_image.save(output_path)
        
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

if __name__ == "__main__":
    main()