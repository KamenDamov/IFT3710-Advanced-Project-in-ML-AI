#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for UNETR2D segmentation model.
Loads a trained model and generates segmentation masks for test images.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Import the model
from unetr2d import UNETR2D

def parse_args():
    parser = argparse.ArgumentParser(description='UNETR2D Segmentation Inference')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output masks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for inference (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference (default: 8)')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of segmentation classes (default: 3)')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels (default: 3)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def get_transform(img_size):
    """Get image transformations for inference"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path, in_channels, num_classes, img_size, device):
    """Load the trained model"""
    model = UNETR2D(
        in_channels=in_channels,
        out_channels=num_classes,
        img_size=(img_size, img_size),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_image_files(directory):
    """Get all image files from directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if os.path.isfile(os.path.join(directory, f)) and 
            any(f.lower().endswith(ext) for ext in image_extensions)]

def process_batch(model, image_batch, device):
    """Process a batch of images"""
    with torch.no_grad():
        input_tensor = torch.stack(image_batch).to(device)
        output = model(input_tensor)
        
        # Get predicted class (argmax over class dimension)
        if output.shape[1] > 1:  # Multi-class segmentation
            predictions = torch.argmax(output, dim=1)
        else:  # Binary segmentation
            predictions = (output > 0).float()
        
        return predictions.cpu()

def save_mask(mask_tensor, output_path, palette=None):
    """Save mask tensor as image"""
    mask_np = mask_tensor.numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask_np)
    
    # Apply color palette if provided
    if palette:
        mask_img = mask_img.convert('P')
        mask_img.putpalette(palette)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the mask
    mask_img.save(output_path)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.in_channels, args.num_classes, args.img_size, device)
    print(f"Model loaded from {args.model_path}")
    
    # Get transform
    transform = get_transform(args.img_size)
    
    # Get image files
    image_files = get_image_files(args.test_dir)
    print(f"Found {len(image_files)} images for inference")
    
    # Define a basic palette for visualization (optional)
    # Adjust based on your number of classes
    if args.num_classes <= 21:
        # Use the PASCAL VOC color palette
        palette = [0, 0, 0]  # Background (black)
        palette += [128, 0, 0]  # Class 1 (red)
        palette += [0, 128, 0]  # Class 2 (green)
        # Add more colors for additional classes as needed
        palette = palette + [0, 0, 0] * (256 - len(palette) // 3)  # Pad to 256 colors
    else:
        palette = None
    
    # Process images in batches
    batch_size = args.batch_size
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    print("Starting inference...")
    for batch_idx in tqdm(range(num_batches)):
        # Get batch files
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        # Prepare batch
        image_batch = []
        for img_path in batch_files:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            image_batch.append(img_tensor)
        
        # Process batch
        batch_predictions = process_batch(model, image_batch, device)
        
        # Save predictions
        for i, img_path in enumerate(batch_files):
            # Generate output path
            img_name = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, os.path.splitext(img_name)[0] + ".png")
            
            # Save mask
            save_mask(batch_predictions[i], output_path, palette)
    
    print(f"Inference completed. Masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()