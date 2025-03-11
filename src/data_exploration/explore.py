from PIL import Image
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
        
# Load a TIFF image file
class Explore:
    def __init(): 
        pass
    def load_tiff_image(self, file_path) -> np.ndarray:
        try:
            imgT = tifffile.imread(file_path)
            tifffile.imshow(imgT, show = True)
            return
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def png_to_np(self, path): 
        return np.array(Image.open(path)) 

explore = Explore()
image_path = "data\preprocessing_outputs\\transformed_images_labels\images\cell_00001.png"
mask_path = "data\preprocessing_outputs\\transformed_images_labels\labels\cell_00001.png"  
image = explore.png_to_np(image_path)
mask = explore.png_to_np(mask_path)
print(image)
print(mask)
n = 3
