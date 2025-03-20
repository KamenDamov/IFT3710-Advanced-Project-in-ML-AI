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


class DoubleConv(nn.Module):
    """
    Double Convolution block:
    (Conv2D -> BatchNorm -> ReLU) twice
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    """
    Downsampling followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """
    Upsampling followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()

        # Use transpose conv if bilinear is False
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample x1
        x1 = self.up(x1)
        
        # Adjust tensor sizes to enable concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 if necessary to match x2 dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for biological cell segmentation
    """
    def __init__(self, in_channels=3, out_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownSample(512, 1024 // factor)
        self.dropout = nn.Dropout(0.5)
        
        # Decoder path
        self.up1 = UpSample(1024, 512 // factor, bilinear)
        self.up2 = UpSample(512, 256 // factor, bilinear)
        self.up3 = UpSample(256, 128 // factor, bilinear)
        self.up4 = UpSample(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Apply sigmoid for outputs between 0 and 1
        return self.sigmoid(logits)


class CellSegmentationDataset(Dataset):
    """
    Dataset class for biological cell segmentation
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask


def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss function for training
    """
    pred = pred.contiguous()
    target = target.contiguous()    
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / 
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """
    Train the UNet model
    
    Args:
        model: UNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model and training history
    """
    print("Training model...")
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'dice_coeff': []
    }
    
    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss (combined BCE and Dice loss)
            bce_loss = criterion(outputs, masks)
            dc_loss = dice_loss(outputs, masks)
            loss = bce_loss + dc_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        dice_coeff = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                # Calculate validation loss
                bce_loss = criterion(outputs, masks)
                dc_loss = dice_loss(outputs, masks)
                loss = bce_loss + dc_loss
                val_loss += loss.item()
                
                # Calculate Dice coefficient (1 - dice loss)
                dice_coeff += 1 - dc_loss.item()
        
        # Log metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        dice_coeff /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['dice_coeff'].append(dice_coeff)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice Coeff: {dice_coeff:.4f}')
    
    return model, history

def predict_and_visualize(model, test_loader, device, num_images=3):
    """
    Make predictions and visualize results
    """
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_images:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            
            # Convert tensors to numpy for visualization
            images = images.cpu().numpy().transpose(0, 2, 3, 1)
            masks = masks.cpu().numpy().transpose(0, 2, 3, 1)
            outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Plot results
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.title('Input Image')
            plt.imshow(images[0])
            plt.axis('off')
            
            plt.subplot(132)
            plt.title('Ground Truth')
            plt.imshow(masks[0])
            plt.axis('off')
            
            plt.subplot(133)
            plt.title('Prediction')
            plt.imshow(outputs[0])
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()


def main():
    # Parameters
    img_size = 256
    batch_size = 8
    epochs = 50
    lr = 0.001
    
    # Directories for your data
    image_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\augmented_dataset\\images"
    mask_dir = "C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\augmented_dataset\\labels"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    full_dataset = CellSegmentationDataset(image_dir, mask_dir, transform=transform)
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=3, bilinear=True)
    
    # Train model
    trained_model, history = train_model(
        model, train_loader, val_loader, device, epochs=epochs, lr=lr
    )
    
    # Save model
    torch.save(trained_model.state_dict(), 'unet_cell_segmentation_augmented_datas.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history['dice_coeff'], label='Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Make predictions and visualize results
    predict_and_visualize(trained_model, test_loader, device)


if __name__ == "__main__":
    main()