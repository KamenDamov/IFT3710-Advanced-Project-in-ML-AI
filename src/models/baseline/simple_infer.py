#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple inference script for UNETR2D segmentation.
Just load the model, generate segmentation masks from test images,
and save the masks to an output directory.
"""

import os
import torch
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

# Import the UNETR2D model
from models.unetr2d import UNETR2D

def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks using UNETR2D')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output masks')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for inference')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of segmentation classes')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNETR2D(
        in_channels=3,
        out_channels=args.num_classes,
        img_size=(args.img_size, args.img_size),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0
    ).to(device)
    
    # Load model weights
    if args.model_path.endswith('.pth'):
        # If model_path points to .pth file
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Try to find the best model in the directory
        best_model_path = os.path.join(args.model_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Could not find model weights at {args.model_path}")
    
    model.eval()
    print(f"Model loaded successfully")
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(args.input_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"Found {len(image_files)} images for inference")
    
    # Process each image
    for image_file in tqdm(image_files):
        # Load and preprocess image
        image_path = os.path.join(args.input_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate prediction
        with torch.no_grad():
            output = model(input_tensor)
            
            # Get segmentation mask
            if args.num_classes > 1:
                # Multi-class segmentation
                mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            else:
                # Binary segmentation
                mask = (output > 0).squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # Save mask
        output_filename = os.path.splitext(image_file)[0] + '.png'
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Convert mask to PIL image and save
        mask_image = Image.fromarray(mask)
        mask_image.save(output_path)
    
    print(f"Inference completed. Masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()