#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for training UNETR2D segmentation model and generating masks.
This script unifies the entire pipeline from training to inference.
"""

import os
import sys
import torch
import argparse
import time
from datetime import datetime
import numpy as np
import shutil
from pathlib import Path

# Import custom modules
from unetr2d import UNETR2D
from dataset_utils import create_dataset, split_dataset, get_transform
from train_utils import train_model
from eval_utils import generate_predictions, evaluate_model
from metrics import compute_metric

def parse_args():
    parser = argparse.ArgumentParser(description='UNETR2D Segmentation Pipeline')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict'],
                        help='Mode of operation: train, test, or predict')
    parser.add_argument('--dataset', type=str, default='baseline',
                        choices=['baseline', 'attention_gan', 'cycle_gan', 'conditionnal_gan', 'all'],
                        help='Dataset to use')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory with input images (used for prediction mode)')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory with ground truth masks (for training)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Base directory for outputs')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory to load model from (for test/predict mode)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and inference')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training and inference')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of segmentation classes')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save models or outputs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    return parser.parse_args()

def get_dataset_paths(dataset_name):
    """Get dataset paths based on dataset name"""
    base_path = "data/preprocessing_outputs"
    
    if dataset_name == 'baseline':
        return {
            'images': f"{base_path}/transformed_images_labels/images",
            'labels': f"{base_path}/transformed_images_labels/labels"
        }
    elif dataset_name == 'attention_gan':
        return {
            'images': f"{base_path}/unified_augmented_data/attention_gan/images",
            'labels': f"{base_path}/unified_augmented_data/attention_gan/labels"
        }
    elif dataset_name == 'cycle_gan':
        return {
            'images': f"{base_path}/unified_augmented_data/cycle_gan/images",
            'labels': f"{base_path}/unified_augmented_data/cycle_gan/labels"
        }
    elif dataset_name == 'conditionnal_gan':
        return {
            'images': f"{base_path}/unified_augmented_data/conditionnal_gan/images",
            'labels': f"{base_path}/unified_augmented_data/conditionnal_gan/labels"
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def setup_directories(base_dir, dataset_name):
    """Setup output directories for a dataset"""
    model_dir = os.path.join(base_dir, 'models', dataset_name)
    pred_dir = os.path.join(base_dir, 'predictions', dataset_name)
    results_dir = os.path.join(base_dir, 'results', dataset_name)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return model_dir, pred_dir, results_dir

def train_pipeline(args):
    """Run the training pipeline"""
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets = ['baseline', 'attention_gan', 'cycle_gan', 'conditionnal_gan']
    else:
        datasets = [args.dataset]
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model configuration
    model_config = {
        'in_channels': 3,
        'out_channels': args.num_classes,
        'img_size': (args.img_size, args.img_size),
        'feature_size': 16,
        'hidden_size': 768,
        'mlp_dim': 3072,
        'num_heads': 12,
        'pos_embed': "perceptron",
        'norm_name': "instance",
        'res_block': True,
        'dropout_rate': 0.0
    }
    
    # Define training configuration
    training_config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'random_seed': 42
    }
    
    # Train on each dataset
    models = {}
    stats = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Training on {dataset_name} dataset")
        print(f"{'='*50}")
        
        # Get dataset paths
        if args.img_dir and args.mask_dir:
            print(f"Using custom paths: {args.img_dir} and {args.mask_dir}")
            dataset_paths = {
                'images': args.img_dir,
                'labels': args.mask_dir
            }
        else:
            dataset_paths = get_dataset_paths(dataset_name)
        
        # Setup directories
        model_dir, pred_dir, results_dir = setup_directories(args.output_dir, dataset_name)
        
        # Split dataset
        print("Creating dataset splits...")
        train_loader, val_loader, test_loader = split_dataset(
            dataset_paths['images'],
            dataset_paths['labels'],
            batch_size=args.batch_size,
            train_ratio=training_config['train_split'],
            val_ratio=training_config['val_split'],
            test_ratio=training_config['test_split'],
            seed=training_config['random_seed'],
            img_size=model_config['img_size']
        )
        
        # Train model
        model, training_stats = train_model(
            model_config,
            training_config,
            train_loader,
            val_loader,
            device,
            model_dir
        )
        
        # Store model and stats
        models[dataset_name] = model
        stats[dataset_name] = training_stats
        
        # Generate predictions on test set
        print("Generating predictions...")
        test_pred_dir = os.path.join(pred_dir, "test_predictions")
        generate_predictions(model, test_loader, test_pred_dir, device)
        
        # Run evaluation metrics
        print("Evaluating model...")
        sys.argv = [
            "compute_metric.py",
            "-g", dataset_paths['labels'],
            "-s", test_pred_dir,
            "--gt_suffix", ".png",
            "--seg_suffix", ".png",
            "-thre", "0.5",
            "-o", results_dir,
            "-n", f"{dataset_name}_results"
        ]
        
        compute_metric.main()
    
    # Compare results if multiple datasets were processed
    if len(datasets) > 1 and not args.no_save:
        print("\nComparing results across datasets...")
        metrics_files = [os.path.join(args.output_dir, 'results', d, f"{d}_results.csv") for d in datasets]
        comparison_path = os.path.join(args.output_dir, "model_comparison.png")
    
    print("\nTraining pipeline completed successfully!")

def test_pipeline(args):
    """Run the testing pipeline"""
    if not args.model_dir:
        raise ValueError("Model directory (--model_dir) must be specified for test mode")
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset paths
    if args.img_dir and args.mask_dir:
        dataset_paths = {
            'images': args.img_dir,
            'labels': args.mask_dir
        }
    else:
        dataset_paths = get_dataset_paths(args.dataset)
    
    # Setup directories
    _, pred_dir, results_dir = setup_directories(args.output_dir, args.dataset)
    
    # Create dataset
    print("Creating dataset...")
    transform = get_transform()
    full_dataset = create_dataset(
        dataset_paths['images'],
        dataset_paths['labels'],
        transform,
        (args.img_size, args.img_size)
    )
    
    # Create dataloader
    loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
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
    
    # Find best model path
    if os.path.isdir(args.model_dir):
        model_path = os.path.join(args.model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            # Try to find latest checkpoint
            checkpoints = [f for f in os.listdir(args.model_dir) if f.startswith("checkpoint_epoch_")]
            if checkpoints:
                checkpoints.sort()
                model_path = os.path.join(args.model_dir, checkpoints[-1])
            else:
                raise FileNotFoundError(f"No model checkpoint found in {args.model_dir}")
    else:
        model_path = args.model_dir
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Generate predictions
    print("Generating predictions...")
    test_pred_dir = os.path.join(pred_dir, "test_predictions")
    generate_predictions(model, loader, test_pred_dir, device)
    
    # Run evaluation metrics
    print("Evaluating model...")
    sys.argv = [
        "compute_metric.py",
        "-g", dataset_paths['labels'],
        "-s", test_pred_dir,
        "--gt_suffix", ".png",
        "--seg_suffix", ".png",
        "-thre", "0.5",
        "-o", results_dir,
        "-n", f"{args.dataset}_results"
    ]
    
    compute_metric.main()
    
    print("\nTesting pipeline completed successfully!")

def predict_pipeline(args):
    """Run the prediction pipeline to generate masks"""
    if not args.model_dir:
        raise ValueError("Model directory (--model_dir) must be specified for predict mode")
    
    if not args.img_dir:
        raise ValueError("Image directory (--img_dir) must be specified for predict mode")
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup directories
    output_dir = args.output_dir
    if not output_dir:
        output_dir = "predictions"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for filename in os.listdir(args.img_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(filename)
    
    if not image_files:
        raise ValueError(f"No valid image files found in {args.img_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
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
    
    # Find best model path
    if os.path.isdir(args.model_dir):
        model_path = os.path.join(args.model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            # Try to find latest checkpoint
            checkpoints = [f for f in os.listdir(args.model_dir) if f.startswith("checkpoint_epoch_")]
            if checkpoints:
                checkpoints.sort()
                model_path = os.path.join(args.model_dir, checkpoints[-1])
            else:
                raise FileNotFoundError(f"No model checkpoint found in {args.model_dir}")
    else:
        model_path = args.model_dir
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create a transform for preprocessing
    transform = get_transform()
    
    # Create a simple dataset with just the images (no labels)
    class ImageOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, files, transform=None, target_size=(256, 256)):
            self.img_dir = img_dir
            self.files = files
            self.transform = transform
            self.target_size = target_size
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            img_name = self.files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size, Image.BILINEAR)
            
            if self.transform:
                image = self.transform(image)
            else:
                # Basic transformation to tensor
                from torchvision import transforms
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            
            return image, img_name
    
    # Create dataset and dataloader
    dataset = ImageOnlyDataset(
        args.img_dir,
        image_files,
        transform,
        (args.img_size, args.img_size)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Generate predictions
    print("Generating segmentation masks...")
    with torch.no_grad():
        for batch_images, batch_filenames in dataloader:
            # Move batch to device
            batch_images = batch_images.to(device)
            
            # Generate predictions
            outputs = model(batch_images)
            
            # Get masks
            if outputs.shape[1] > 1:  # Multi-class segmentation
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:  # Binary segmentation
                preds = (outputs > 0.5).float().cpu().numpy()
            
            # Save each prediction
            for i, filename in enumerate(batch_filenames):
                pred = preds[i].astype(np.uint8)
                
                from PIL import Image
                pred_img = Image.fromarray(pred)
                output_path = os.path.join(output_dir, filename)
                pred_img.save(output_path)
    
    print(f"\nPrediction completed! Generated {len(image_files)} segmentation masks in {output_dir}")

def main():
    args = parse_args()
    
    start_time = time.time()
    
    if args.mode == 'train':
        train_pipeline(args)
    elif args.mode == 'test':
        test_pipeline(args)
    elif args.mode == 'predict':
        predict_pipeline(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main()