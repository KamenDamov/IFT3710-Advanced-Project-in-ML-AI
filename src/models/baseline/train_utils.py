#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities for UNETR2D segmentation model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from unetr2d import UNETR2D

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for images, masks, _ in tqdm(dataloader, desc="Training"):
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
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cpu or cuda)
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validation"):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

def train_model(model_config, training_config, train_loader, val_loader, device, model_dir):
    """
    Train the UNETR2D model.
    
    Args:
        model_config (dict): Model configuration parameters
        training_config (dict): Training configuration parameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cpu or cuda)
        model_dir (str): Directory to save model checkpoints
        
    Returns:
        tuple: (trained model, training statistics)
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize model
    model = UNETR2D(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        img_size=model_config['img_size'],
        feature_size=model_config['feature_size'],
        hidden_size=model_config['hidden_size'],
        mlp_dim=model_config['mlp_dim'],
        num_heads=model_config['num_heads'],
        pos_embed=model_config['pos_embed'],
        norm_name=model_config['norm_name'],
        res_block=model_config['res_block'],
        dropout_rate=model_config['dropout_rate']
    ).to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Initialize tracking variables
    num_epochs = training_config['num_epochs']
    best_val_loss = float('inf')
    training_stats = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Start training
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store statistics
        training_stats['train_loss'].append(train_loss)
        training_stats['val_loss'].append(val_loss)
        training_stats['lr'].append(current_lr)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for return
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth")))
    
    return model, training_stats