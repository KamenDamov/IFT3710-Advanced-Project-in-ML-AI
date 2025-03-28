import torch
import os
from PIL import Image 
import numpy as np
import cv2
 
class Infer:
        
    def generate_pseudo_masks(self, model, input_dir, output_dir, image_gen=False):
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith(".tiff") or filename.endswith(".tif"):
                tiff_image = Image.open(os.path.join(input_dir, filename))
                png_filename = filename.rsplit(".", 1)[0] + ".png"
                tiff_image.save(os.path.join(input_dir, png_filename), format="PNG")
                print(f"Converted {filename} to {png_filename}")            
                filename = png_filename
                
            elif not filename.endswith((".png", ".jpg", ".jpeg")):
                continue
                
            print(f"Processing {filename}")
            image_path = os.path.join(input_dir, filename)
            
            # Load the image
            if image_gen:
                # For grayscale images - ensure 4D tensor [B,C,H,W]
                image = np.array(Image.open(image_path).convert("L")).astype(np.float32)
                image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            else: 
                # For RGB images - ensure 4D tensor [B,C,H,W]
                image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
                image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # [H,W,C] -> [C,H,W] -> [1,C,H,W]
            
            # Run inference
            with torch.no_grad():
                # Move image to the same device as the model
                device = next(model.parameters()).device
                image = image.to(device)
                
                # Forward pass
                generated = model(image)
            
            # Process the output
            generated = generated.squeeze().cpu().detach().numpy()
            
            if len(generated.shape) == 3:  # If RGB output
                # If all channels contain similar information, take the mean
                # and then apply thresholding
                generated_single_channel = np.mean(generated, axis=0)
                
                # Apply threshold to create binary mask with white objects on black background
                threshold = 0.5  # Adjust based on your model's output range
                generated_single_channel = (generated_single_channel > threshold).astype(np.uint8) * 255
            else:
                # Already single channel
                threshold = 0.5  # Adjust based on your model's output range
                generated_single_channel = (generated > threshold).astype(np.uint8) * 255

            # Save output - no need for additional scaling
            mask_output_path = os.path.join(output_dir, filename)
            cv2.imwrite(mask_output_path, generated_single_channel)
            
            print(f"Generated pseudo-mask saved: {mask_output_path}")
        
