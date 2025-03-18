import torch
import os
from PIL import Image 
import numpy as np
import cv2
 
class Infer:
    
    def generate_pseudo_masks(self, model, input_dir, output_dir, image_gen = True):
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
            print("Here")
            image_path = os.path.join(input_dir, filename)
            if image_gen:
                image = torch.tensor(np.array(Image.open(image_path).convert("L")).astype(np.float32)).unsqueeze(0)
            else: 
                image = torch.tensor(np.array(Image.open(image_path).convert("RGB")).astype(np.float32))
                image = image.permute(2, 0, 1)
            with torch.no_grad():
                generated = model.netG(image.to(model.device))  # Run Pix2Pix generator
            
            # Convert tensor to image
            generated = generated.squeeze().cpu().detach().numpy()
            if image_gen:
                generated = np.transpose(generated, (1, 2, 0))
                generated = (generated + 1) * 127.5  # Convert from [-1,1] to [0,255]
                generated = np.clip(generated, 0, 255).astype(np.uint8)
            else:
                generated = (generated + 1) * 127.5  # Convert from [-1,1] to [0,255]
                generated = np.clip(generated, 0, 255).astype(np.uint8)

            # Save pseudo-mask
            mask_output_path = os.path.join(output_dir, filename)
            cv2.imwrite(mask_output_path, generated)

            print(f"Generated pseudo-mask saved: {mask_output_path}")

    
