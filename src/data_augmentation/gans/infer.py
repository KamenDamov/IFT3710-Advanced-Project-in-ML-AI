import torch
import os
from PIL import Image 
import numpy as np
import cv2
from metrics import compute_metric
 
class Infer:
    def __init__(): 
        pass
    
    def generate_pseudo_masks(model, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue
            
            image_path = os.path.join(input_dir, filename)
            image = torch.tensor(np.array(Image.open(image_path).convert("RGB")).astype(np.float32))
            image = image.permute(2, 0, 1)
            with torch.no_grad():
                generated = model.netG(image.to(model.device))  # Run Pix2Pix generator
            
            # Convert tensor to image
            generated = generated.squeeze().cpu().detach().numpy()
            generated = (generated + 1) * 127.5  # Convert from [-1,1] to [0,255]
            generated = np.clip(generated, 0, 255).astype(np.uint8)

            # Save pseudo-mask
            mask_output_path = os.path.join(output_dir, filename)
            cv2.imwrite(mask_output_path, generated)

            print(f"Generated pseudo-mask saved: {mask_output_path}")
        
    def generate_performance_metrics(path): 
        pass

    
