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

explore = Explore()
file_path_tiff_label = 'C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\Training-labeled\\Training-labeled\\labels\\cell_00001_label.tiff'
file_path_png_label = 'C:\\Users\\kamen\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\notebooks\\preprocessing_outputs\\labels\\cell_00073_label.png'
image = explore.load_tiff_image(file_path_tiff_label)
