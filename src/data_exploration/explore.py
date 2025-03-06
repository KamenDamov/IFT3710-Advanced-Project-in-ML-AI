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
            img = imgT.convert("RGB")
            pixels = list(img.getdata())
            num_pixels = len(pixels)
            avg_color = tuple(sum(col) // num_pixels for col in zip(*pixels))
            print(f"Average pixel color: {avg_color}")
            image = cv2.imread(file_path)
            image = np.asarray(image,dtype = np.float64)
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                pixels = list(img.getdata())
                num_pixels = len(pixels)
                avg_color = tuple(sum(col) // num_pixels for col in zip(*pixels))
                print(f"Average pixel color: {avg_color}")
                return img
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def load_png_image(self, file_path):
        img=mpimg.imread(file_path)
        imgplot = plt.imshow(img)
        plt.show()

explore = Explore()
file_path_tiff_label = 'C:\\Users\\kamen\\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\data\\Training-labeled\\Training-labeled\\labels\\cell_00001_label.tiff'
file_path_png_label = 'C:\\Users\\kamen\Dev\\School\\H25\\IFT3710\\IFT3710-Advanced-Project-in-ML-AI\\notebooks\\preprocessing_outputs\\labels\\cell_00073_label.png'
#image = explore.load_tiff_image(file_path_tiff)
image = explore.load_png_image(file_path_png_label)