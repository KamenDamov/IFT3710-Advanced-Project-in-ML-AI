import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
import os
from torchvision.utils import save_image

# Base GAN class
class BaseGAN(nn.Module):
    def __init__(self):
        super(BaseGAN, self).__init__()
        self.generator = None
        self.discriminator = None
    
    def forward(self, z):
        return self.generator(z)
    
    def generate(self, z):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(z)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=256, img_channels=1):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),  # 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16 x 128 x 128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, img_channels, kernel_size=4, stride=2, padding=1, bias=False),  # 1 x 256 x 256
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: 1 x 256 x 256
            nn.Conv2d(img_channels, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16 x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),  # 1 x 1 x 1
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.net(img).view(-1, 1)
# Full GAN Model
class GANModel(BaseGAN):
    def __init__(self, z_dim, img_channels):
        super(GANModel, self).__init__()
        self.generator = Generator(z_dim, img_channels)
        self.discriminator = Discriminator(img_channels)
        self.criterion = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Data Loader
class CellDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx])  # Implement load_image()
        if self.transform:
            image = self.transform(image)
        return image

    def load_image(self, path):
        image = Image.open(path).convert("L")  # Convert to grayscale
        return image    

def create_dataloader(image_folder, batch_size=32, img_size=64):
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CellDataset(image_paths, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Trainer
class Trainer:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
    
    def train(self, epochs, sample_dir="samples"):
        # Create directory for samples if it doesn't exist
        os.makedirs(sample_dir, exist_ok=True)
        
        for epoch in range(epochs):
            for i, real_images in enumerate(self.dataloader):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Correct noise dimension: batch_size x z_dim x 1 x 1
                z = torch.randn(batch_size, 256, 1, 1, device=self.device)
                fake_images = self.model.generator(z)
                
                # Train Discriminator
                self.model.optimizer_d.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # Get discriminator outputs and reshape
                d_real_output = self.model.discriminator(real_images)
                d_fake_output = self.model.discriminator(fake_images.detach())
                
                d_real_loss = self.model.criterion(d_real_output, real_labels)
                d_fake_loss = self.model.criterion(d_fake_output, fake_labels)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.model.optimizer_d.step()
                
                # Train Generator
                self.model.optimizer_g.zero_grad()
                g_loss = self.model.criterion(self.model.discriminator(fake_images), real_labels)
                g_loss.backward()
                self.model.optimizer_g.step()
                
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(self.dataloader)} "
                        f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
            
            # Save model checkpoint every epoch
            torch.save(self.model.state_dict(), f"gan_256x256_epoch_{epoch+1}.pth")
            
            # Generate and save a sample image at the end of each epoch
            self.save_sample_image(epoch, sample_dir)
            
            print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
        
        # Save final model
        torch.save(self.model.state_dict(), "gan_256x256_final.pth")

    def save_sample_image(self, epoch, sample_dir="samples", proceed_training=False):
        """Generate and save a sample image at the end of each epoch"""
        self.model.generator.eval()
        with torch.no_grad():
            # Generate a fixed noise vector for comparison across epochs
            fixed_noise = torch.randn(1, 256, 1, 1, device=self.device)
            fake_image = self.model.generator(fixed_noise).cpu().detach()
            
            # Denormalize image (from [-1, 1] to [0, 1] range)
            fake_image = (fake_image + 1) / 2.0
            
            # Convert tensor to PIL image and save
            save_image(fake_image, f"{sample_dir}/sample_epoch_{epoch+1}.png")
            
            # Alternative method using PIL directly
            # from PIL import Image
            # import numpy as np
            # img_array = fake_image[0, 0].numpy() * 255
            # img_array = img_array.astype(np.uint8)
            # img = Image.fromarray(img_array)
            # img.save(f"{sample_dir}/sample_epoch_{epoch+1}.png")
        if proceed_training: 
            self.model.generator.train()


# Evaluator
class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def generate_samples(self, num_samples):
        # Correct noise dimension: num_samples x z_dim x 1 x 1
        z = torch.randn(num_samples, 256, 1, 1, device=self.device)
        samples = self.model.generate(z)
        return samples.cpu().detach()
    
    def save_sample_image(self, sample, epoch, i,  save_dir="generated_samples"):
        os.makedirs(save_dir, exist_ok=True)
        save_image(sample, f"{save_dir}/sample_{i+1}_epoch_{epoch}.png")
    
    def evaluate(self, real_samples, fake_samples):
        # Implement evaluation metrics like IoU, Dice Coefficient, etc.
        pass