#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities for UNETR2D segmentation model.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on (cpu or cuda)
        
    Returns:
        tuple: (average loss, metrics dict)
    """
    model.eval()
    running_loss = 0.0
    
    # Initialize metrics
    total_pixels = 0
    correct_pixels = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # Get predictions
            if outputs.shape[1] > 1:  # Multi-class segmentation
                preds = torch.argmax(outputs, dim=1)
            else:  # Binary segmentation
                preds = (outputs > 0.5).float()
            
            # Calculate accuracy
            correct = (preds == masks).float()
            correct_pixels += correct.sum().item()
            total_pixels += masks.numel()
            
            # Calculate per-class metrics
            for c in range(outputs.shape[1]):  # For each class
                class_mask = (masks == c)
                if class_mask.sum() > 0:  # If there are pixels of this class
                    class_correct_pixels = ((preds == c) & class_mask).float().sum().item()
                    class_total_pixels = class_mask.float().sum().item()
                    
                    # Update class metrics
                    if c not in class_correct:
                        class_correct[c] = 0
                        class_total[c] = 0
                    
                    class_correct[c] += class_correct_pixels
                    class_total[c] += class_total_pixels
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    
    # Calculate overall accuracy
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for c in class_total:
        class_accuracy[c] = class_correct[c] / class_total[c] if class_total[c] > 0 else 0
    
    # Return metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'mean_class_accuracy': np.mean(list(class_accuracy.values()))
    }
    
    return avg_loss, metrics

def generate_predictions(model, dataloader, output_dir, device):
    """
    Generate and save predictions from the model.
    
    Args:
        model: The model to use for prediction
        dataloader: DataLoader for the data to predict
        output_dir: Directory to save predictions
        device: Device to run prediction on (cpu or cuda)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, _, filenames in tqdm(dataloader, desc="Generating predictions"):
            # Move data to device
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Convert to segmentation masks
            if outputs.shape[1] > 1:  # Multi-class segmentation
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:  # Binary segmentation
                preds = (outputs > 0.5).float().cpu().numpy()
            
            # Save predictions
            for i, filename in enumerate(filenames):
                pred = preds[i].astype(np.uint8)
                pred_img = Image.fromarray(pred)
                pred_img.save(os.path.join(output_dir, filename))

def predict_single_image(model, image_path, output_path=None, device=None):
    """
    Generate a prediction for a single image.
    
    Args:
        model: The model to use for prediction
        image_path: Path to input image
        output_path: Path to save output mask (optional)
        device: Device to run prediction on (optional)
        
    Returns:
        numpy.ndarray: Predicted segmentation mask
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to model's expected input size
    input_size = model.img_size
    image = image.resize(input_size, Image.BILINEAR)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        
        # Get mask
        if output.shape[1] > 1:  # Multi-class segmentation
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        else:  # Binary segmentation
            pred = (output > 0.5).float().squeeze().cpu().numpy()
    
    # Save prediction if output path is provided
    if output_path is not None:
        pred_img = Image.fromarray(pred.astype(np.uint8))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pred_img.save(output_path)
    
    return pred

def calculate_metrics(pred_masks, gt_masks):
    """
    Calculate metrics between predicted masks and ground truth.
    
    Args:
        pred_masks: Predicted segmentation masks (N, H, W)
        gt_masks: Ground truth masks (N, H, W)
        
    Returns:
        dict: Dictionary with metrics
    """
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()
    
    # Flatten masks
    pred_flat = pred_masks.reshape(-1)
    gt_flat = gt_masks.reshape(-1)
    
    # Calculate pixel accuracy
    pixel_acc = np.mean(pred_flat == gt_flat)
    
    # Get unique classes
    classes = np.unique(gt_flat)
    
    # Calculate IoU for each class
    iou_per_class = {}
    dice_per_class = {}
    
    for c in classes:
        pred_c = (pred_flat == c)
        gt_c = (gt_flat == c)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        
        # Calculate IoU
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        iou_per_class[int(c)] = iou
        
        # Calculate Dice coefficient
        if pred_c.sum() + gt_c.sum() > 0:
            dice = 2 * intersection / (pred_c.sum() + gt_c.sum())
        else:
            dice = 0.0
        
        dice_per_class[int(c)] = dice
    
    # Calculate mean metrics
    mean_iou = np.mean(list(iou_per_class.values()))
    mean_dice = np.mean(list(dice_per_class.values()))
    
    metrics = {
        'pixel_accuracy': pixel_acc,
        'iou_per_class': iou_per_class,
        'mean_iou': mean_iou,
        'dice_per_class': dice_per_class,
        'mean_dice': mean_dice
    }
    
    return metrics