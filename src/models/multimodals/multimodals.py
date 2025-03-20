import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import random

class CLIPSegmentationModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super(CLIPSegmentationModel, self).__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.feature_dim = self.clip_model.config.projection_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, images, input_ids, attention_mask):
        #with torch.no_grad():
        vision_features = self.clip_model.get_image_features(images)
        text_features = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        combined_features = torch.cat([vision_features, text_features], dim=1)
        fused = self.fusion(combined_features)
        reshaped = fused.view(-1, 256, 16, 16)
        mask = self.decoder(reshaped)
        return F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

class CellSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, caption_file, clip_processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.clip_processor = clip_processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        with open(caption_file, 'r') as f:
            self.captions_data = json.load(f)
        self.data = [(os.path.basename(item["image"]), f"{os.path.splitext(os.path.basename(item['image']))[0]}.png", random.choice(list(item["captions"].values()))) for item in self.captions_data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, mask_name, caption = self.data[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert('L')
        text_inputs = self.clip_processor(text=[caption], return_tensors="pt", padding=True, truncation=True, max_length=77)
        return {
            'pixel_values': self.transform(img),
            'input_ids': text_inputs.input_ids[0],
            'attention_mask': text_inputs.attention_mask[0],
            'mask': transforms.ToTensor()(mask),
            'caption': caption
        }

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Padding des input_ids et des attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'masks': masks,
        'captions': captions
    }

def dice_loss(pred, target):
    smooth = 1.0
    pred_flat, target_flat = pred.view(-1), target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = sum((criterion(output := model(batch['pixel_values'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device)), batch['masks'].to(device)) + dice_loss(output, batch['masks'].to(device))).item() for batch in train_loader) / len(train_loader)
        
        model.eval()
        with torch.no_grad():
            val_loss = sum((criterion(output := model(batch['pixel_values'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device)), batch['masks'].to(device)) + dice_loss(output, batch['masks'].to(device))).item() for batch in val_loader) / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
    
    model.load_state_dict(torch.load("best_model.pth"))
    return model

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'masks': masks,
        'captions': captions
    }

def dice_loss(pred, target):
    smooth = 1.0
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return 1 - ((2. * intersection + smooth) / 
                (pred_flat.sum() + target_flat.sum() + smooth))

# def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track best model
    best_val_loss = float('inf')
    best_model_path = "best_cell_segmentation_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader.dataset:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # Compute loss (combination of BCE and Dice loss for better segmentation)
            bce_loss = criterion(outputs, masks)
            dice = dice_loss(outputs, masks)
            loss = bce_loss + dice
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                masks = batch['masks'].to(device)
                
                outputs = model(pixel_values, input_ids, attention_mask)
                
                bce_loss = criterion(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = bce_loss + dice
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    return model

def visualize_predictions(model, test_loader, save_dir="predictions", num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            masks = batch['masks'].to(device)
            captions = batch['captions']
            
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # Convert tensors to numpy for visualization
            img = pixel_values[0].cpu().permute(1, 2, 0).numpy()
            # Normalize image for display
            img = (img - img.min()) / (img.max() - img.min())
            
            true_mask = masks[0, 0].cpu().numpy()
            pred_mask = outputs[0, 0].cpu().numpy()
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(true_mask, cmap='gray')
            axes[1].set_title('True Mask')
            axes[1].axis('off')
            
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.suptitle(f"Caption: {captions[0]}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"prediction_{i}.png"))
            plt.close()

def main():
    # Configuration
    img_dir = "C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\transformed_images_labels\\images"
    mask_dir = "C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\data\\preprocessing_outputs\\transformed_images_labels\\labels"
    caption_file = "C:\\Users\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\data\\preprocessing_outputs\\captions_results.json"# "../../../data/preprocessing_outputs/captions_results.json"
    batch_size = 8
    num_epochs = 10
    
    # Initialize CLIP processor
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create dataset
    dataset = CellSegmentationDataset(img_dir, mask_dir, caption_file, clip_processor)
    print(f"Dataset size: {len(dataset)}")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Initialize model
    model = CLIPSegmentationModel()
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Save final model
    torch.save(trained_model.state_dict(), "cell_segmentation_clip_model.pth")
    
    # Visualize some predictions
    visualize_predictions(trained_model, val_loader)
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()