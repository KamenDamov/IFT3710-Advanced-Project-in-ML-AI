#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for dataset preparation and handling for UNETR2D segmentation.
"""

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class SegmentationDataset(Dataset):
    """Dataset class for semantic segmentation tasks."""
    
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on images.
            target_size (tuple): Size to resize images to.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all valid image files
        self.image_files = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_path = os.path.join(mask_dir, filename)
                if os.path.exists(mask_path):
                    self.image_files.append(filename)
        
        # Sort for reproducibility
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, img_name)
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Resize to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)  # NEAREST to preserve mask values
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Basic transformation to tensor
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask, img_name

def create_dataset(image_dir, mask_dir, transform=None, target_size=(256, 256)):
    """
    Create a dataset instance for semantic segmentation.
    
    Args:
        image_dir (str): Directory with images.
        mask_dir (str): Directory with masks.
        transform (callable, optional): Transform to apply to images.
        target_size (tuple): Size to resize images to.
        
    Returns:
        SegmentationDataset: Dataset for semantic segmentation.
    """
    return SegmentationDataset(image_dir, mask_dir, transform, target_size)

def get_transform():
    """Get standard transformation for segmentation images."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class FilteredDataset(Dataset):
    """Dataset that filters an original dataset to include only specified files."""
    
    def __init__(self, original_dataset, file_indices):
        """
        Args:
            original_dataset: Original dataset to filter
            file_indices: List of indices to include
        """
        self.dataset = original_dataset
        self.indices = file_indices
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def split_dataset(image_dir, mask_dir, batch_size=8, train_ratio=0.7, val_ratio=0.15, 
                 test_ratio=0.15, seed=42, img_size=(256, 256)):
    """
    Split a dataset into train, validation, and test sets and create DataLoaders.
    
    Args:
        image_dir (str): Directory with images.
        mask_dir (str): Directory with masks.
        batch_size (int): Batch size for DataLoaders.
        train_ratio (float): Ratio of data for training.
        val_ratio (float): Ratio of data for validation.
        test_ratio (float): Ratio of data for testing.
        seed (int): Random seed for reproducibility.
        img_size (tuple): Image size for resizing.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset
    transform = get_transform()
    full_dataset = create_dataset(image_dir, mask_dir, transform, img_size)
    
    # Get all indices
    indices = list(range(len(full_dataset)))
    
    # Split the data
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=seed)
    
    # Adjust validation ratio
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio_adjusted, random_state=seed)
    
    # Create filtered datasets
    train_dataset = FilteredDataset(full_dataset, train_idx)
    val_dataset = FilteredDataset(full_dataset, val_idx)
    test_dataset = FilteredDataset(full_dataset, test_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset split: Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_loader, val_loader, test_loader