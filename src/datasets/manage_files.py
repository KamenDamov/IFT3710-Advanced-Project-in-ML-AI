import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile as tif
import pandas as pd
from skimage import io, color
import PIL.Image as Image
import numpy as np
import colorsys
import zipfile
import os
import cv2

VISIBLE_TYPES = [".bmp", ".png", ".jpg", ".jpeg"]
TENSOR_TYPES = [".tif", ".tiff"]
IMAGE_TYPES = VISIBLE_TYPES + TENSOR_TYPES
MISC_TYPES = [".md", ".zip", ".txt", ".csv", ".py", ".json", ""]
FILE_TYPES = IMAGE_TYPES + MISC_TYPES

IMAGE = 'Image'
MASK = 'Mask'
SYNTHETIC = 'Synthetic'

class BaseFileSet:
    def __init__(self, root):
        self.root = root

    def crosscheck(self, dataroot):
        for filepath, category in self.enumerate(dataroot):
            if category is None:
                print("Unknown category:", filepath)
            if category != IMAGE:
                continue
            for maskpath, type in self.mask_filepath(filepath):
                if not os.path.exists(dataroot + maskpath):
                    print("Missing", type, ":", maskpath, "from", filepath)
                crosscat = self.categorize(maskpath)
                if crosscat != type:
                    print("Wrong category:", maskpath, "as", crosscat, "!=", type, "from", filepath)
            
    def unpack(self, dataroot):
        unzip_dataset(dataroot, self.root + "/")

    def blacklist(self, filepath):
        return False
    
    def signature(self, filepath):
        (dims, channels, type) = tensor_signature(self.load(filepath))
        category = self.categorize(filepath)
        return (dims, channels, type, category)
    
    def enumerate(self, folder):
        for filepath in enumerate_dataset(folder, self.root + "/"):
            dirpath, name, ext = split_filepath(filepath)
            if ext not in IMAGE_TYPES:
                continue
            if self.blacklist(filepath):
                continue
            category = self.categorize(filepath)
            yield filepath, category
    
    def mask_filepath(self, filepath):
        return []
    
    def categorize(self, filepath):
        return None
    
    def load(self, filepath):
        return load_image(filepath)

def split_filepath(filepath):
    dirpath, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    separator = '/' if (not dirpath or dirpath[-1] != '/') else ''
    return dirpath + separator, name, ext

def unzip_archive(root, filepath, needs_offset=[]):
    dirpath, name, ext = split_filepath(filepath)
    print("Inspecting archive: ", filepath)
    with zipfile.ZipFile(root + filepath, 'r') as zip_ref:
        # Top level folders
        offset = (name + "/" if name in needs_offset else "")
        folders = set(dirpath + offset + zipname.split('/')[0] for zipname in zip_ref.namelist())
        missing = [folder for folder in folders if not os.path.exists(root + folder)]
        if missing:
            print("Unzipping archive: ", missing)
            zip_ref.extractall(root + dirpath + offset)
            return missing
        return []

def list_dataset(root, folder = '/'):
    files_by_type = {type:set() for type in FILE_TYPES}
    for filepath in enumerate_dataset(root, folder):
        dirpath, name, ext = split_filepath(filepath)
        files_by_type[ext].add(filepath)
    return files_by_type

def unzip_dataset(root, folder, needs_offset=[]):
    for filepath in enumerate_dataset(root, folder):
        unzip_datafile(root, filepath, needs_offset)

def unzip_datafile(root, filepath, needs_offset=[]):
    dirpath, name, ext = split_filepath(filepath)
    if ext != ".zip":
        return
    for unzipped in unzip_archive(root, filepath, needs_offset):
        if not os.path.exists(root + unzipped):
            print("!WARNING! Archive did not produce folder: ", root + unzipped)
        elif os.path.isdir(root + unzipped):
            unzip_dataset(root, unzipped + "/", needs_offset)
        else:
            unzip_datafile(root, unzipped, needs_offset)

def enumerate_dataset(root, folder):
    #print("Enumerating folder: ", folder)
    for filename in os.listdir(root + folder):
        filepath = folder + filename
        yield filepath
        dirpath, name, ext = split_filepath(filepath)
        if os.path.isdir(root + filepath):
            for fullpath in enumerate_dataset(root, filepath + "/"):
                yield fullpath

def tensor_signature(tensor):
    dims = len(tensor.shape)
    channels = 1 if dims < 3 else tensor.shape[-1]
    return (dims, channels, tensor.dtype)

def load_image(img_path):
    dirpath, name, ext = split_filepath(img_path)
    if ext in TENSOR_TYPES:
        return tif.imread(img_path)
    elif ext in VISIBLE_TYPES:
        return io.imread(img_path)

def save_image(img_path, image):
    dirpath, name, ext = split_filepath(img_path)
    if ext in TENSOR_TYPES:
        return tif.imwrite(img_path, image)
    elif ext in VISIBLE_TYPES:
        return io.imsave(img_path, image, check_contrast=False)
    elif ext == ".csv":
        return image.to_csv(img_path)

def target_file(filepath, ext):
    dirpath, name, _ = split_filepath(filepath)
    return dirpath + name + ext

def safely_process(log, process, overwrite=False):
    def wrapper(source, target):
        try:
            if overwrite or not os.path.exists(target):
                dirpath, _, _ = split_filepath(target)
                os.makedirs(dirpath, exist_ok=True)
                result = process(source, target)
                if result is not None:
                    save_image(target, result)
        except Exception as e:
            print(e)
            log.append(source + " -> " + target)
    return wrapper
