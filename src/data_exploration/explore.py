import matplotlib.pyplot as plt
import tifffile as tif
import pandas as pd
import PIL.Image as Image
import numpy as np
import colorsys
import zipfile
import os
import cv2

IMAGE_TYPES = [".bmp", ".png", ".tif", ".tiff"]
MISC_TYPES = [".md", ".zip", ".txt", ".csv", ".py", ""]

LABELED = 'Labeled'
MASK = 'Mask'
UNLABELED = 'Unlabeled'
SYNTHETIC = 'Synthetic'

def split_filepath(filepath):
    dirpath, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    separator = '/' if (not dirpath or dirpath[-1] != '/') else ''
    return dirpath + separator, name, ext

def unzip_archive(root, filepath):
    dirpath, name, ext = split_filepath(filepath)
    print("Inspecting archive: ", filepath)
    with zipfile.ZipFile(root + filepath, 'r') as zip_ref:
        # Top level folders
        folders = [dirpath + zipname for zipname in zip_ref.namelist() if '/' not in zipname[:-1]]
        missing = [folder for folder in folders if not os.path.exists(root + folder)]
        if missing:
            print("Unzipping archive: ", missing)
            zip_ref.extractall(root + dirpath)
            return missing
        return []

def list_dataset(root, folder = '/'):
    files_by_type = {type:set() for type in (IMAGE_TYPES + MISC_TYPES)}
    for filepath in enumerate_dataset(root, folder):
        dirpath, name, ext = split_filepath(filepath)
        files_by_type[ext].add(filepath)
    return files_by_type

def unzip_dataset(root, folder):
    for filepath in enumerate_dataset(root, folder):
        dirpath, name, ext = split_filepath(filepath)
        if ext != ".zip":
            continue
        for unzipped in unzip_archive(root, filepath):
            if os.path.exists(root + unzipped):
                unzip_dataset(root, unzipped)
            else:
                print("!WARNING! Archive did not produce folder: ", root + unzipped)

def enumerate_dataset(root, folder):
    print("Enumerating folder: ", folder)
    for filename in os.listdir(root + folder):
        filepath = folder + filename
        yield filepath
        dirpath, name, ext = split_filepath(filepath)
        if not ext:
            for fullpath in enumerate_dataset(root, filepath + "/"):
                yield fullpath

class ZenodoNeurIPS:
    def __init__(self, root, category = None):
        self.root = root
        self.category = category

    def label_patterns(self, category):
        if category == MASK:
            yield ("/Public/images/", "/Public/labels/")
            yield ("/Public/WSI/", "/Public/WSI-labels/")
            yield ("/Training-labeled/images/", "/Training-labeled/labels/")
            yield ("/Tuning/images/", "/Tuning/labels/")
        if category == SYNTHETIC:
            yield ("/Hidden/images/", "/Hidden/osilab_seg/")
            yield ("/Public/images/", "/Public/1st_osilab_seg/")
            yield ("/Public/WSI/", "/Public/osilab_seg_WSI/")

    def mask_filepath(self, filepath):
        folder, name, ext = split_filepath(filepath)
        folder = self.mask_folder(folder)
        return (folder + name + "_label.tiff") if folder else None

    def mask_folder(self, folder):
        for (img, mask) in self.label_patterns(self.category):
            if img in folder:
                return folder.replace(img, mask)
        return None

    def categorize(self, dirpath):
        for (img, mask) in self.label_patterns(MASK):
            if mask in dirpath:
                return MASK
            elif img in dirpath:
                return LABELED
        for (img, mask) in self.label_patterns(SYNTHETIC):
            if mask in dirpath:
                return SYNTHETIC
        return UNLABELED

def dataset_frame(root, matcher):
    files_by_type = list_dataset(root, matcher.root + '/')
    assoc = collect_datamap(root, matcher, files_by_type)
    numbers = list(collect_dataset(root, assoc))
    return pd.DataFrame(numbers, columns = ["Path", "Mask", "Width", "Height", "Objects", "Background"]).set_index("Path")

def collect_datamap(root, matcher, files_by_type):
    assoc = {}
    for ext in IMAGE_TYPES:
        for filepath in files_by_type[ext]:
            maskpath = matcher.mask_filepath(filepath)
            if not maskpath:
                continue
            elif os.path.exists(root + maskpath):
                assoc[filepath] = maskpath
            else:
                print("Missing mask: ", root, maskpath, filepath)
    return assoc

def collect_dataset(root, assoc):
    expected = len(assoc)
    for index, (img_path, datapath) in enumerate(assoc.items()):
        print(index, "/", expected)
        # Get global statistics
        imgT = tif.imread(root + datapath)
        background = (imgT == 0).sum()
        yield {"Path":img_path, "Mask":datapath, "Width": imgT.shape[1], "Height":imgT.shape[0], "Objects": imgT.max(), "Background": background}

def merge_lists(compare, merge, listA, listB):
    merged = [0] * (len(listA) + len(listB))
    indexM, indexA, indexB = 0, 0, 0
    while indexA < len(listA) and indexB < len(listB):
        cmp = compare(listA[indexA], listB[indexB])
        if cmp < 0:
            merged[indexM] = listA[indexA]
            indexM += 1
            indexA += 1
        elif cmp > 0:
            merged[indexM] = listB[indexB]
            indexM += 1
            indexB += 1
        else:
            merged[indexM] = merge(listA[indexA], listB[indexB])
            indexM += 1
            indexA += 1
            indexB += 1
            merged.pop()
    if indexA < len(listA):
        merged[indexM:] = listA[indexA:]
    if indexB < len(listB):
        merged[indexM:] = listB[indexB:]
    return merged

def mask_frame_leaf(tensor, bounds):
    tensor = tensor[bounds.top:bounds.bottom, bounds.left:bounds.right]
    objectIDs = np.unique(tensor)
    return [detectObject(tensor, bounds, id) for id in objectIDs]

def mask_frame(root, mask_path):
    tensor = tif.imread(root + mask_path)
    bounds = BoundingBox()
    bounds.right = tensor.shape[1]
    bounds.bottom = tensor.shape[0]
    print(bounds.width(), "x", bounds.height())
    print(tensor.max(), "objects")
    objects = mask_frame_branch(tensor, bounds)
    for object in objects:
        normalizeObject(object)
    return pd.DataFrame(objects, columns = ["ID", "X", "Y", "Left", "Right", "Top", "Bottom", "Area"]).set_index("ID")

def mask_frame_branch(tensor, bounds):
    if bounds.small():
        return mask_frame_leaf(tensor, bounds)
    splitA, splitB = bounds.split()
    objectsA, objectsB = mask_frame_branch(tensor, splitA), mask_frame_branch(tensor, splitB)
    merged = merge_lists(compareObjects, mergeObjects, objectsA, objectsB)
    return merged

class BoundingBox:
    def __init__(self):
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
    
    def copy(self):
        bounds = BoundingBox()
        bounds.left = self.left
        bounds.right = self.right
        bounds.top = self.top
        bounds.bottom = self.bottom
        return bounds
    
    def width(self):
        return self.right - self.left
    
    def height(self):
        return self.bottom - self.top
    
    def small(self):
        return self.width() < 128 and self.height() < 128
    
    def split(self):
        boundsA = self.copy()
        boundsB = self.copy()
        if self.height() < self.width():
            #split = self.left + int(np.exp2(np.ceil(np.log2(self.width()) - 1)))
            split = self.left + self.width() // 2
            boundsA.right = split
            boundsB.left = split
        else:
            #split = self.top + int(np.exp2(np.ceil(np.log2(self.height()) - 1)))
            split = self.top + self.height() // 2
            boundsA.bottom = split
            boundsB.top = split
        return (boundsA, boundsB)

def normalizeObject(object):
    object["X"] = np.int32(np.round(object["X"]))
    object["Y"] = np.int32(np.round(object["Y"]))
    return object

def detectObject(tensor, bounds, id):
    # Get local statistics
    rows, cols = np.where(tensor == id)
    x = bounds.left + cols.mean()
    y = bounds.top + rows.mean()
    left = bounds.left + cols.min()
    right = bounds.left + cols.max() + 1
    top = bounds.top + rows.min()
    bottom = bounds.top + rows.max() + 1
    area = len(rows)
    return {"ID": np.int32(id), "X": x, "Y": y, "Left": left, "Right": right, "Top": top, "Bottom": bottom, "Area": area}

def compareObjects(objectA, objectB):
    return objectA["ID"] - objectB["ID"]

def mergeObjects(objectA, objectB):
    assert objectA["ID"] == objectB["ID"]
    area = objectA["Area"] + objectB["Area"]
    x = (objectA["X"] * objectA["Area"] + objectB["X"] * objectB["Area"]) / area
    y = (objectA["Y"] * objectA["Area"] + objectB["Y"] * objectB["Area"]) / area
    left = min(objectA["Left"], objectB["Left"])
    right = max(objectA["Right"], objectB["Right"])
    top = min(objectA["Top"], objectB["Top"])
    bottom = max(objectA["Bottom"], objectB["Bottom"])
    return {"ID": objectA["ID"], "X": x, "Y": y, "Left":left, "Right":right, "Top": top, "Bottom":bottom, "Area": area}

def save_maskframes(dataroot, df):
    root = dataroot + "/raw"
    store = dataroot + "/processed"
    for index, mask_path in enumerate(df["Mask"]):
        print(index, "/", len(df))
        folder, name, ext = split_filepath(mask_path)
        maskframe = mask_frame(root, mask_path)
        target = store + folder + name
        os.makedirs(store + folder, exist_ok=True)
        maskframe.to_csv(target + ".csv")

# Save black-white mask
def save_bw_mask(root, store, datapath):
    imgT = tif.imread(root + datapath)
    im = Image.fromarray((imgT != 0).astype('uint8')*255)
    folder, name, ext = split_filepath(datapath)
    target = store + folder
    os.makedirs(target, exist_ok=True)
    maskfile = target + name + ".png"
    im.save(maskfile)

# TOO SLOW
# Save hue-vector mask
def save_hue_mask(root, store, datapath, df):
    imgT = tif.imread(root + datapath)
    imgTC = np.zeros((imgT.shape[0], imgT.shape[1], 3), dtype=np.float32)
    for index, row in df[1:].iterrows():
        i = row['ID']
        bounds = [row['Top'], row['Bottom'], row['Left'], row['Right']]
        middle = np.array([row['Y'], row['X']])
        rows, cols = np.where(imgT[bounds[0]:bounds[1], bounds[2]:bounds[3]] == i)
        for r, c in zip(rows, cols):
            location = np.array([r, c])
            vector = (location - middle)/np.array([bounds[1] - bounds[0], bounds[3] - bounds[2]])
            # Get euclidian norm
            norm = np.linalg.norm(vector)
            angle = np.arctan2(vector[1], vector[0])  # Calculate the angle in radians
            hue = (angle + np.pi) / (2 * np.pi)  # Normalize angle to [0, 1] for hue
            rgb = colorsys.hsv_to_rgb(hue, norm, 1.0)  # Convert HSV to RGB
            imgTC[r, c] = rgb
    print(imgTC.min(), imgTC.max())
    im = Image.fromarray((imgTC * 255).astype('uint8'), mode="RGB")
    folder, name, ext = split_filepath(datapath)
    target = store + folder
    os.makedirs(target, exist_ok=True)
    maskfile = target + name + ".vect.png"
    im.save(maskfile)

def enumerate_frames(dataroot):
    for filename in os.listdir(dataroot):
        dirpath, name, ext = split_filepath(filename)
        if ext == ".csv":
            dataframe = pd.read_csv(dataroot + dirpath + filename)
            yield name, dataframe

def preprocess_masks(dataroot, df, color = False):
    rawroot = dataroot + "/raw"
    procroot = dataroot + "/processed"
    for index, datapath in enumerate(df["Mask"]):
        print(index, "/", len(df))
        if color:
            save_hue_mask(rawroot, procroot, datapath)
        else:
            save_bw_mask(rawroot, procroot, datapath)

def preprocess_images(dataroot, df):
    for index, filepath in enumerate(df["Path"]):
        print(index, "/", len(df), filepath)
        folder, name, ext = split_filepath(filepath)
        img = cv2.imread(dataroot + "/raw" + filepath)
        target = dataroot + "/processed" + folder
        os.makedirs(target, exist_ok=True)
        cv2.imwrite(target + name + ".png", img)

def preprocess_dataset(dataroot):
    rawroot = dataroot + "/raw"
    zenodo = ZenodoNeurIPS('/zenodo')
    unzip_dataset(rawroot, zenodo.root + "/")
    for category, label in [(MASK, ".labels"), (SYNTHETIC, ".synth")]:
        zenodo.category = category
        data_map = dataset_frame(rawroot, zenodo)
        data_map.to_csv(dataroot + zenodo.root + label + ".csv")
        
