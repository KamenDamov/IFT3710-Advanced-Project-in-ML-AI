"""
Visualization script for MONAI transformations applied to medical images.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import monai
from monai.data import PILReader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    SpatialPadd,
    RandSpatialCropd,
    RandAxisFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    RandZoomd,
    EnsureTyped,
    Compose,
)

def visualize_transform(data_item, transform, title):
    """Apply a transform and visualize before/after results"""
    try:
        # Create a copy to avoid modifying the original
        data_copy = {k: v for k, v in data_item.items()}
        
        # Apply transform
        result = transform(data_copy)
        
        # Create figure
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 3, figure=fig)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        if "img" in data_item:
            if isinstance(data_item["img"], str):
                ax1.text(0.5, 0.5, f"File path: {os.path.basename(data_item['img'])}", 
                         ha='center', va='center', wrap=True)
            elif isinstance(data_item["img"], np.ndarray):
                # Handle different dimensions
                if data_item["img"].ndim == 3 and data_item["img"].shape[2] == 3:  # HWC
                    ax1.imshow(data_item["img"])
                elif data_item["img"].ndim == 3 and data_item["img"].shape[0] == 3:  # CHW
                    ax1.imshow(np.transpose(data_item["img"], (1, 2, 0)))
                else:
                    ax1.imshow(data_item["img"], cmap='gray')
            elif isinstance(data_item["img"], torch.Tensor):
                img_np = data_item["img"].numpy()
                if img_np.ndim == 3 and img_np.shape[0] == 3:  # CHW
                    ax1.imshow(np.transpose(img_np, (1, 2, 0)))
                else:
                    ax1.imshow(img_np, cmap='gray')
        else:
            ax1.text(0.5, 0.5, "No image data", ha='center', va='center')
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Original label
        ax2 = fig.add_subplot(gs[0, 1])
        if "label" in data_item:
            if isinstance(data_item["label"], str):
                ax2.text(0.5, 0.5, f"File path: {os.path.basename(data_item['label'])}", 
                         ha='center', va='center', wrap=True)
            elif isinstance(data_item["label"], np.ndarray):
                if data_item["label"].ndim == 3 and data_item["label"].shape[0] == 1:  # CHW
                    ax2.imshow(data_item["label"][0], cmap='viridis')
                else:
                    ax2.imshow(data_item["label"], cmap='viridis')
            elif isinstance(data_item["label"], torch.Tensor):
                label_np = data_item["label"].numpy()
                if label_np.ndim == 3 and label_np.shape[0] == 1:  # CHW
                    ax2.imshow(label_np[0], cmap='viridis')
                else:
                    ax2.imshow(label_np, cmap='viridis')
        else:
            ax2.text(0.5, 0.5, "No label data", ha='center', va='center')
        ax2.set_title("Original Label")
        ax2.axis('off')
        
        # Transformed image
        ax3 = fig.add_subplot(gs[1, 0])
        if "img" in result:
            # Convert to numpy array if it's a torch tensor
            if isinstance(result["img"], torch.Tensor):
                img_data = result["img"].detach().cpu().numpy()
            else:
                img_data = result["img"]
            
            # Handle different dimensions
            if img_data.ndim == 3 and img_data.shape[0] == 3:  # CHW
                ax3.imshow(np.transpose(img_data, (1, 2, 0)))
            else:
                ax3.imshow(img_data, cmap='gray')
        else:
            ax3.text(0.5, 0.5, "No transformed image", ha='center', va='center')
        ax3.set_title("Transformed Image")
        ax3.axis('off')
        
        # Transformed label
        ax4 = fig.add_subplot(gs[1, 1])
        if "label" in result:
            # Convert to numpy array if it's a torch tensor
            if isinstance(result["label"], torch.Tensor):
                label_data = result["label"].detach().cpu().numpy()
            else:
                label_data = result["label"]
            
            # Handle different dimensions
            if label_data.ndim == 3 and label_data.shape[0] == 1:  # CHW
                ax4.imshow(label_data[0], cmap='viridis')
            else:
                ax4.imshow(label_data, cmap='viridis')
        else:
            ax4.text(0.5, 0.5, "No transformed label", ha='center', va='center')
        ax4.set_title("Transformed Label")
        ax4.axis('off')
        
        # Info panel
        ax5 = fig.add_subplot(gs[:, 2])
        ax5.axis('off')
        
        # Display image shape information
        info_text = f"Transform: {title}\n\n"
        
        if "img" in data_item:
            if isinstance(data_item["img"], np.ndarray):
                info_text += f"Original Image Shape: {data_item['img'].shape}\n"
            elif isinstance(data_item["img"], torch.Tensor):
                info_text += f"Original Image Shape: {tuple(data_item['img'].shape)}\n"
            elif isinstance(data_item["img"], str):
                info_text += f"Original Image: File path\n"
                
        if "label" in data_item:
            if isinstance(data_item["label"], np.ndarray):
                info_text += f"Original Label Shape: {data_item['label'].shape}\n"
            elif isinstance(data_item["label"], torch.Tensor):
                info_text += f"Original Label Shape: {tuple(data_item['label'].shape)}\n"
            elif isinstance(data_item["label"], str):
                info_text += f"Original Label: File path\n"
        
        if "img" in result:
            if isinstance(result["img"], np.ndarray):
                img_data = result["img"]
                info_text += f"Transformed Image Shape: {img_data.shape}\n"
                info_text += f"Image Value Range: [{img_data.min():.4f}, {img_data.max():.4f}]\n"
            elif isinstance(result["img"], torch.Tensor):
                img_data = result["img"].detach().cpu().numpy()
                info_text += f"Transformed Image Shape: {tuple(result['img'].shape)}\n"
                info_text += f"Image Value Range: [{img_data.min():.4f}, {img_data.max():.4f}]\n"
        
        if "label" in result:
            if isinstance(result["label"], np.ndarray):
                label_data = result["label"]
                info_text += f"Transformed Label Shape: {label_data.shape}\n"
                unique_values = np.unique(label_data)
                info_text += f"Label Unique Values: {unique_values}\n"
            elif isinstance(result["label"], torch.Tensor):
                label_data = result["label"].detach().cpu().numpy()
                info_text += f"Transformed Label Shape: {tuple(result['label'].shape)}\n"
                unique_values = np.unique(label_data)
                info_text += f"Label Unique Values: {unique_values}\n"
        
        ax5.text(0.1, 0.9, info_text, va='top', ha='left', wrap=True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        return result
    
    except Exception as e:
        print(f"Error visualizing '{title}' transform:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return data_item  # Return the original item to continue the visualization
    

def main():
    # Set parameters
    input_size = 256
    
    # Ask user for path to a sample image and label
    # data_path = input("Enter the path to your data directory (containing 'images' and 'labels' folders): ")
    img_path = "C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\notebooks\\preprocessing_outputs\\images"
    gt_path = "C:\\Users\\Samir\\Documents\\GitHub\\IFT3710-Advanced-Project-in-ML-AI\\notebooks\\preprocessing_outputs\\labels"
    
    img_names = sorted(os.listdir(img_path))
    
    if not img_names:
        print("No images found in the specified directory.")
        return
    
    print("Available images:")
    for i, name in enumerate(img_names):
        print(f"{i+1}. {name}")
    
    selection = int(input("Select image number to visualize: ")) - 1
    if selection < 0 or selection >= len(img_names):
        print("Invalid selection.")
        return
    
    selected_img = img_names[selection]
    selected_gt = selected_img.split(".")[0] + "_label.png"
    
    # Check if the label file exists
    label_path = os.path.join(gt_path, selected_gt)
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        proceed = input("Do you want to continue without a label? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Create a data item
    data_item = {
        "img": os.path.join(img_path, selected_img),
        "label": os.path.join(gt_path, selected_gt)
    }
    
    # Define individual transforms with proper parameters
    transforms = [
        ("Load Image", LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8)),
        ("Ensure Channel First", EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1)),
        ("Scale Intensity", ScaleIntensityd(keys=["img"], allow_missing_keys=True)),
        ("Spatial Padding", SpatialPadd(keys=["img", "label"], spatial_size=input_size)),
        ("Random Spatial Crop", RandSpatialCropd(keys=["img", "label"], roi_size=input_size, random_size=False)),
        ("Random Axis Flip", RandAxisFlipd(keys=["img", "label"], prob=1.0, spatial_axes=[0, 1])),
        ("Random Rotate 90", RandRotate90d(keys=["img", "label"], prob=1.0, spatial_axes=[0, 1])),
        ("Random Gaussian Noise", RandGaussianNoised(keys=["img"], prob=1.0, mean=0, std=0.1)),
        ("Random Adjust Contrast", RandAdjustContrastd(keys=["img"], prob=1.0, gamma=1.5)),
        # Fix for the error - provide sigma_x as a tuple/list
        ("Random Gaussian Smooth", RandGaussianSmoothd(keys=["img"], prob=1.0, sigma_x=(1.5, 2.0))),
        ("Random Histogram Shift", RandHistogramShiftd(keys=["img"], prob=1.0, num_control_points=3)),
        ("Random Zoom", RandZoomd(
            keys=["img", "label"], 
            prob=1.0, 
            min_zoom=0.8, 
            max_zoom=1.5, 
            mode=("area", "nearest"))),
        ("Ensure Type", EnsureTyped(keys=["img", "label"]))
    ]
    
    # Visualize each transform step by step
    current_data = data_item
    for name, transform in transforms:
        print(f"Applying: {name}")
        current_data = visualize_transform(current_data, transform, name)
    
    # Optional: Show the full transformation pipeline at once
    try:
        print("Applying full transformation pipeline...")
        full_transforms = Compose([t for _, t in transforms])
        visualize_transform(data_item, full_transforms, "Complete Pipeline")
    except Exception as e:
        print(f"Error applying full pipeline: {str(e)}")
    

if __name__ == "__main__":
    main()