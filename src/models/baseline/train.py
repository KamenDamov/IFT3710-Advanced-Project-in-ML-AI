#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for UNETR2D segmentation model.
"""

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time

# Import custom modules
from models.unetr2d import UNETR2D
from datasets.segmentation_dataset import SegmentationDataset
from utils.transforms import get_training_transforms, get_validation_transforms
from utils.losses import get_loss_function
from utils.metrics import MetricTracker

def parse_args():
    parser = argparse.ArgumentParser(description='UNETR2D Segmentation Training')
    parser.add_argument('--train_img_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--train_mask_dir', type=str, required=True, help='Directory with training masks')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Directory with validation images')
    parser.add_argument('--val_mask_dir', type=str, required=True, help='Directory with validation masks')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes (default: 3)')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels (default: 3)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--loss', type=str, default='combined', help='Loss function: ce, dice, focal, combined (default: combined)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping patience (default: 20)')
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device, metric_tracker):
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update loss
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Get predictions and update metrics
        if outputs.shape[1] > 1:  # Multi-class segmentation
            preds = torch.argmax(outputs, dim=1)
        else:  # Binary segmentation
            preds = (outputs > 0).float()
        
        metric_tracker.update(preds, masks)
    
    # Calculate average loss
    avg_loss = epoch_loss / len(dataloader)
    
    # Calculate metrics
    metrics = metric_tracker.get_results()
    
    return avg_loss, metrics

def validate(model, dataloader, criterion, device, metric_tracker):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Update loss
            val_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Get predictions and update metrics
            if outputs.shape[1] > 1:  # Multi-class segmentation
                preds = torch.argmax(outputs, dim=1)
            else:  # Binary segmentation
                preds = (outputs > 0).float()
            
            metric_tracker.update(preds, masks)
    
    # Calculate average loss
    avg_loss = val_loss / len(dataloader)
    
    # Calculate metrics
    metrics = metric_tracker.get_results()
    
    return avg_loss, metrics

def save_checkpoint(model, optimizer, epoch, loss, metrics, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best performance
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
    
    return checkpoint_path

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create tensorboard writer
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    # Define transforms
    train_transform = get_training_transforms() if args.augmentation else get_validation_transforms()
    val_transform = get_validation_transforms()
    
    # Create datasets
    train_dataset = SegmentationDataset(
        image_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        transform=train_transform,
        target_size=(args.img_size, args.img_size),
        augmentation=args.augmentation,
        num_classes=args.num_classes
    )
    
    val_dataset = SegmentationDataset(
        image_dir=args.val_img_dir,
        mask_dir=args.val_mask_dir,
        transform=val_transform,
        target_size=(args.img_size, args.img_size),
        augmentation=False,
        num_classes=args.num_classes
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = UNETR2D(
        in_channels=args.in_channels,
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
    
    # Define loss function
    criterion = get_loss_function(args.loss, ignore_index=255)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['loss']
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Start training
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Reset metric trackers
        train_metric_tracker = MetricTracker(args.num_classes)
        val_metric_tracker = MetricTracker(args.num_classes)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metric_tracker
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, val_metric_tracker
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, mIoU: {train_metrics['mean_iou']:.4f}, Dice: {train_metrics['mean_dice']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, mIoU: {val_metrics['mean_iou']:.4f}, Dice: {val_metrics['mean_dice']:.4f}")
        
        # Save metrics to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mIoU/train', train_metrics['mean_iou'], epoch)
        writer.add_scalar('mIoU/val', val_metrics['mean_iou'], epoch)
        writer.add_scalar('Dice/train', train_metrics['mean_dice'], epoch)
        writer.add_scalar('Dice/val', val_metrics['mean_dice'], epoch)
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        
        if is_best:
            best_val_loss = val_loss
            early_stopping_counter = 0
            print(f"New best model! Val Loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping: {early_stopping_counter}/{args.early_stopping}")
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, val_loss, val_metrics,
            args.output_dir, is_best
        )
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Early stopping
        if early_stopping_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()