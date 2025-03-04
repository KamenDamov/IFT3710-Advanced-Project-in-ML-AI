from PIL import Image
import cv2
import numpy as np
import tifffile
import matplotlib as plt
# Load a TIFF image file
def load_tiff_image(file_path):
    try:
        imgT = tifffile.imread(file_path)
        print(np.average(imgT))
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

# Example usage
if __name__ == "__main__":
    file_path = './data/Training-labeled/Training-labeled/labels/cell_00008_label.tiff'
    image = load_tiff_image(file_path)