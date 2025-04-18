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
from src.manage_files import *
from src.livecell import LiveCellSet

### Dataset sources ###
# MEDIAR
# https://github.com/Lee-Gihun/MEDIAR
#
# NeurIPS Competition
# https://zenodo.org/records/10719375
# 
# OmniPose
# https://github.com/kevinjohncutler/omnipose
# https://www.nature.com/articles/s41592-022-01639-4#data-availability
# https://osf.io/xmury/files/osfstorage
# 
# CellPose
# https://www.cellpose.org/dataset
# 
# DataScienceBowl
# https://www.kaggle.com/competitions/data-science-bowl-2018/data
#
###

class ScienceBowlSet(BaseFileSet):
    def mask_filepath(self, filepath):
        if "/images" in filepath:
            yield (filepath.replace("/images", "/labels"), MASK)
    
    def categorize(self, filepath):
        if "/masks" in filepath:
            return MASK
        if "/images" in filepath:
            return LABELED
        return UNLABELED

class OmniPoseSet(BaseFileSet):
    def blacklist(self, filepath):
        return ("/worm" in filepath) \
            or ("_flows" in filepath)
    
    def mask_filepath(self, filepath):
        folder, name, ext = split_filepath(filepath)
        yield (folder + name + "_masks" + ext), MASK
    
    def categorize(self, filepath):
        if "_masks" in filepath:
            return MASK
        return LABELED

class CellposeSet(BaseFileSet):
    def mask_filepath(self, filepath):
        if ("img" in filepath):
            yield filepath.replace("img", "masks"), MASK
    
    def categorize(self, filepath):
        if "masks" in filepath:
            return MASK
        elif "img" in filepath:
            return LABELED
        return UNLABELED
    
    def blacklist(self, filepath):
        # This mask is corrupted
        return "train_cyto2/758" in filepath
    
    def load(self, filepath):
        category = self.categorize(filepath)
        if category == LABELED:
            image = BaseFileSet.load(self, filepath)
            image = np.flip(image, axis=2) # BGR -> RGB
            channels = image.sum(axis=(0, 1))
            # The Cellpose dataset contains grayscale images that are green-coded
            if channels.sum() == channels[1]:
                return image.sum(axis=2)
            return image
        if category == MASK:
            # The Cellpose dataset contains those outliers (65535 = 2^16) as background for some reason
            mask = BaseFileSet.load(self, filepath)
            mask[mask == (2 ** 16 - 1)] = 0
            return mask

class ZenodoNeurIPS(BaseFileSet):
    def blacklist(self, filepath):
        # Don't process unlabeled images for now
        return "release-part1" in filepath \
            or "train-unlabeled-part2" in filepath \
            or "unlabeled_cell_00504" in filepath # File is cut off

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
        for folder, category in self.mask_folder(folder):
            yield (folder + name + "_label.tiff"), category

    def mask_folder(self, folder):
        for category in [MASK, SYNTHETIC]:
            for (img, mask) in self.label_patterns(category):
                if img in folder:
                    yield folder.replace(img, mask), category
    
    def categorize(self, dirpath):
        for category in [MASK, SYNTHETIC]:
            for (img, mask) in self.label_patterns(category):
                if mask in dirpath:
                    return category
                elif img in dirpath:
                    return LABELED
        return UNLABELED

def collect_datamap(matcher, files_by_type):
    assoc = {MASK:{}, SYNTHETIC:{}}
    for ext in IMAGE_TYPES:
        for filepath in files_by_type[ext]:
            category = matcher.categorize(filepath)
            if category in [MASK, SYNTHETIC]:
                for maskpath, category in matcher.mask_filepath(filepath):
                    assoc[category][filepath] = maskpath
    return assoc

def collect_dataset(root, assoc):
    for category in [MASK, SYNTHETIC]:
        for (imagepath, maskpath) in tqdm(assoc[category].items()):
            if not os.path.exists(root + maskpath):
                print("Missing mask: ", root, maskpath, imagepath)
            # Get global statistics
            imgT = DataSet.load_raw(None, root + maskpath)
            background = (imgT == 0).sum()
            num_objects = imgT.max()
            synthetic = (category == SYNTHETIC)
            yield {"Path":imagepath, "Mask":maskpath, "Synthetic": synthetic, "Width": imgT.shape[1], "Height":imgT.shape[0], "Objects": num_objects, "Background": background}

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

def mask_frame(tensor):
    bounds = BoundingBox()
    bounds.right = tensor.shape[1]
    bounds.bottom = tensor.shape[0]
    #print(bounds.width(), "x", bounds.height())
    #print(tensor.max(), "objects")
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

def save_maskframe(mask_path, frame_path):
    tensor = DataSet.load_raw(None, mask_path)
    maskframe = mask_frame(tensor)
    maskframe.to_csv(frame_path)

# Save black-white mask
def save_bw_mask(source, target):
    imgT = DataSet.load_raw(None, source)
    im = Image.fromarray((imgT != 0).astype('uint8')*255)
    im.save(target)

# Save grayscale mask
def save_gray_mask(source, target):
    imgT = DataSet.load_raw(None, source)
    num_objects = imgT.max()
    im = Image.fromarray((imgT / num_objects * 255).astype('uint8'))
    im.save(target)

# TOO SLOW
# Save hue-vector mask
def save_hue_mask(root, store, datapath, df):
    imgT = DataSet.load_raw(None, root + datapath)
    imgTC = np.zeros((imgT.shape[0], imgT.shape[1], 3), dtype=np.float32)
    for index, row in df[1:].iterrows():
        i = row['ID']
        bounds = [row['Top'], row['Bottom'], row['Left'], row['Right']]
        middle = np.array([row['Y'], row['X']])
        rows, cols = np.where(imgT[bounds[0]:bounds[1], bounds[2]:bounds[3]] == i)
        rows += bounds[0]
        cols += bounds[2]
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

def save_clean_image(source, target):
    img = DataSet.load_raw(None, source)
    img = Image.fromarray(img.astype('uint8'))
    img.save(target)
    #cv2.imwrite(target, img)

def prepare_metaframe(dataroot, target_path):
    dataset = [ZenodoNeurIPS('/neurips'), CellposeSet("/cellpose"), OmniPoseSet("/omnipose"), LiveCellSet("/livecell"), ScienceBowlSet("/sciencebowl")]
    data_map = dataset_frame(dataroot, dataset)
    data_map.to_csv(target_path)
    return data_map

def dataset_frame(root, matchers):
    numbers = []
    for matcher in matchers:
        unzip_dataset(root, matcher.root + "/")
        files_by_type = list_dataset(root, matcher.root + '/')
        assoc = collect_datamap(matcher, files_by_type)
        numbers += collect_dataset(root, assoc)
    frame = pd.DataFrame(numbers, columns = ["Path", "Mask", "Width", "Height", "Objects", "Background", "Synthetic"])
    frame = frame.set_index("Path")
    return frame.sort_index()

class DataSet:
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.meta_frame = self.dataroot + "/dataset.labels.csv"
        self.df = self.prepare_frame()

    def __str__(self):
        return self.dataroot
    
    def __len__(self):
        return len(self.df)
    
    def prepare_frame(self):
        rawroot = self.dataroot + "/unify"
        safely_process([], prepare_metaframe)(rawroot, self.meta_frame)
        return pd.read_csv(self.meta_frame)

    def __iter__(self):
        for index in range(len(self.df)):
            row = self.df.iloc[index]
            sample = DataSample(self.dataroot, row)
            if not self.blacklist(sample):
                yield sample

    def blacklist(self, sample):
        return sample.df["Synthetic"] \
            or ("train_cyto2/758" in sample.df["Path"]) \
            or ("WSI" in sample.df["Path"])
    
    # Load a well-formed tensor from the raw image
    def load_raw(self, filepath):
        return load_image(filepath)

class DataSample:
    def __init__(self, dataroot, df):
        self.dataroot = dataroot
        self.df = df

        self.name = split_filepath(df["Path"])[1]
        self.init_paths(df["Path"], df["Mask"])
        self.width = df.get('Width')
        self.height = df.get('Height')
    
    def __str__(self):
        return str(self.df)
    
    def prepare_frame(self):
        safely_process([], save_maskframe)(self.raw_mask, self.meta_frame)

    def labels(self):
        self.prepare_frame(self)
        return DataLabels(self.meta_frame)
    
    def init_paths(self, image_path, mask_path):
        self.raw_image = self.dataroot + "/raw" + image_path
        self.raw_mask = self.dataroot + "/raw" + mask_path
        self.clean_image = self.dataroot + "/processed" + target_file(image_path, ".png")
        self.bw_mask = self.dataroot + "/processed" + target_file(mask_path, ".bin.png")
        self.gray_mask = self.dataroot + "/processed" + target_file(mask_path, ".gray.png")
        self.meta_frame = self.dataroot + "/processed" + target_file(mask_path, ".csv")
        self.normal_image = self.dataroot + "/preprocessing_outputs/normalized_data" + target_file(image_path, ".png")
        self.normal_mask = self.dataroot + "/preprocessing_outputs/normalized_data" + target_file(mask_path, ".png")
    
    def transform_image(self, index):
        return self.dataroot + "/preprocessing_outputs/transformed_images_labels/images" + f"/{self.name}.{index}.png"

    def transform_mask(self, index):
        return self.dataroot + "/preprocessing_outputs/transformed_images_labels/labels" + f"/{self.name}.{index}.png"


class DataLabels:
    def __init__(self, meta_path):
        self.df = pd.read_csv(meta_path)
        self.width = self.df['Right'].max()
        self.height = self.df['Bottom'].max()
    
    def __str__(self):
        return str(self.df)
    
    def __len__(self):
        return len(self.df)
    
    def lerp(self, a, b):
        return a + b - (a * b)
    
    def relscalar(self, box):
        [left, top, right, bottom] = box
        sx = right - left
        sy = bottom - top
        x = left / (1 - sx) if sx != 1 else 0.5
        y = top / (1 - sy) if sy != 1 else 0.5
        return [x, y, sx, sy]
    
    def relbox(self, scalar):
        [x, y, sx, sy] = scalar
        # x = left/(1 - s)
        left = x * (1 - sx)
        right = left + sx
        top = y * (1 - sy)
        bottom = top + sy
        return [left, top, right, bottom]

    def absbox(self, relbox):
        [left, top, right, bottom] = relbox
        left = int(np.rint(left * self.width))
        right = int(np.rint(right * self.width))
        top = int(np.rint(top * self.height))
        bottom = int(np.rint(bottom * self.height))
        return [left, top, right, bottom]

    def dictbox(self, box):
        [left, top, right, bottom] = box
        width = right - left
        height = bottom - top
        area = width * height
        return { 'Left': left, 'Top': top, 'Right': right, 'Bottom': bottom, 'Width': width, 'Height': height, 'Area': area }
    
    def flatbox(self, df):
        return [df['Left']/self.width, df['Top']/self.height, df['Right']/self.width, df['Bottom']/self.height]
    
    def randscalar(self, random):
        x = random.beta(0.5, 0.5)
        y = random.beta(0.5, 0.5)
        sx = random.beta(2, 2)
        sy = sx #random.beta(2, 1)
        return [x, y, sx, sy]
    
    def randboxsmart(self, random):
        scalar = self.randscalar(random)
        relbox = self.relbox(scalar)
        absbox = self.absbox(relbox)
        return self.dictbox(absbox)
    
    def randobject(self, random):
        weights = self.df['Area'].sum() / self.df['Area']
        weights = list(weights / weights.sum())
        # sample an integer according to given weights
        return random.choice(len(self.df), p=weights)
    
    def select_slices(self, random):
        choice = self.randobject(random)
        if not choice:
            return self.randboxsmart(random)
        dictbox = self.df.iloc[choice]
        relbox = self.flatbox(dictbox)
        [x, y, sx, sy] = self.relscalar(relbox)
        [_, _, rx, ry] = self.randscalar(random)
        scalar = [x, y, self.lerp(sx, rx), self.lerp(sy, ry)]
        relbox = self.relbox(scalar)
        absbox = self.absbox(relbox)
        return self.dictbox(absbox)

def sanity_check(dataset, sample):
    image = dataset.load_raw(sample.raw_image)
    #print(image.shape, sample.raw_image)
    if len(image.shape) == 3:
        assert image.shape[2] == 3
        channels = image.sum(axis=(0, 1))
        for channel in range(3):
            if channels[channel] == 0:
                print("Found empty channel: ", image.shape, channel, sample.raw_image)
    tensor = dataset.load_raw(sample.raw_mask)
    dense = (len(np.unique(tensor)) == tensor.max()+1)
    if 65535 in tensor:
        print("Found 65535 in: ", sample.raw_mask)
    if not dense:
        print("Found non-dense mask: ", sample.raw_mask)
    df = DataLabels(sample.meta_frame).df
    width = df['Right'].max()
    height = df['Bottom'].max()
    assert tensor.shape[0] == height
    assert tensor.shape[1] == width
    area = df['Area'].sum()
    assert (width * height) == area
    assert area != 0
    for index in range(len(df)):
        bdf = df.iloc[index]
        sanity_check_box(width, height, bdf)

def sanity_check_box(width, height, bdf):
    bwidth = bdf['Right'] - bdf['Left']
    bheight = bdf['Bottom'] - bdf['Top']
    barea = bwidth * bheight
    #print(width, height, bdf)
    assert 0 <= bwidth <= width
    assert 0 <= bheight <= height
    assert 0 <= bdf['Area'] <= barea
    assert 0 <= bdf['Left'] <= width
    assert 0 <= bdf['Right'] <= width
    assert 0 <= bdf['Top'] <= height
    assert 0 <= bdf['Bottom'] <= height

def check_signatures(dataroot, datasets):
    for dataset in datasets:
        files = list(dataset.enumerate(dataroot))
        print(dataset.root, ":", len(files), "files")
        signatures = set(dataset.signature(dataroot + filepath) for filepath in tqdm(files))
        print("=", signatures)

def unify_dataset(dataroot, dataset):
    files = list(dataset.enumerate(dataroot + "/raw"))
    for filepath in tqdm(files, desc="Unifying dataset " + dataset.root):
        folder, name, ext = split_filepath(filepath)
        source = dataroot + "/raw" + filepath
        target_type = ext #".tiff" if dataset.categorize(filepath) == MASK else ".png"
        target = dataroot + "/unify" + folder + name + target_type
        safely_process([], unify_file(dataset))(source, target)

def unify_file(dataset):
    # The dataset knows how to standardize its own image files
    def unify(source, target):
        mask = unify_load(dataset, source)
        save_image(target, mask)
    return unify

def upper_bound(tensor):
    max = tensor.max()
    for bytes in [0, 1, 2, 4, 8]:
        bits = bytes << 3
        if max < (1 << bits):
            return bits

def sanitize(tensor, rescale):
    assert 0 <= tensor.min(), "Tensor contains negative values"
    bits = upper_bound(tensor)
    if tensor.shape[-1] == 4:  # RGBA
        tensor = color.rgba2rgb(tensor)
    if rescale and (bits != 8):
        scaled = tensor * ((1 << 8)/(1 << bits))
        return scaled.astype('uint8')
    return tensor.astype('uint' + str(bits))

def unify_load(dataset, filepath):
    image = dataset.load(filepath)
    category = dataset.categorize(filepath)
    rescale = category not in [MASK, SYNTHETIC]
    sanitized = sanitize(image, rescale)
    if (image.dtype != sanitized.dtype):
        #print("WARNING: " + str(image.dtype) + " -> " + str(sanitized.dtype) + ("(rescale)" if rescale else "") + ", " + filepath)
        pass
    return sanitized

if __name__ == "__main__":
    dataroot = "./data"
    datasets = [ZenodoNeurIPS("/neurips"), CellposeSet("/cellpose"), OmniPoseSet("/omnipose"), LiveCellSet(), ScienceBowlSet("/sciencebowl")]
    #unzip_dataset(dataroot, "/raw/")
    #check_signatures(dataroot + "/raw", datasets[0:1])

    #image = load_image("./data/unify/neurips/Testing/Hidden/osilab_seg/TestHidden_379_label.tiff")
    #image = load_image("./data/raw/neurips/Tuning/labels/cell_00074_label.tiff")
    #print(image.shape, image.dtype, image.min(), image.max())
    #image = ZenodoNeurIPS("/neurips").load("./data/raw/neurips/Tuning/labels/cell_00074_label.tiff")
    #print(image.shape, image.dtype, image.min(), image.max())
    #image = load_image("./data/raw/neurips/Training-labeled/images/cell_00315.tiff")
    #print(image.shape, image.dtype, image.min(), image.max())
    #print(np.dtype('<u2') == np.uint16)
    #print(mask_frame(image))

    unify_dataset(dataroot, datasets[0])
    unify_dataset(dataroot, datasets[1])
    unify_dataset(dataroot, datasets[2])
    unify_dataset(dataroot, datasets[3])

if False: #__name__ == "__main__":
    dataset = DataSet("./data")
    for sample in tqdm(dataset, desc="Preparing metadata frames"):
        sample.prepare_frame()
        sanity_check(dataset, sample)
        # Also save human-readable mask images, for debugging
        safely_process([], save_clean_image)(sample.raw_image, sample.clean_image)
        safely_process([], save_bw_mask)(sample.raw_mask, sample.bw_mask)
        safely_process([], save_gray_mask)(sample.raw_mask, sample.gray_mask)

"""
/neurips : 3158 files
= {(3, 3, dtype('uint8'), 'Labeled'), (2, 1, dtype('<u2'), 'Labeled'), (2, 1, dtype('int32'), 'Labeled'), (2, 1, dtype('float64'), 'Labeled'), (3, 3, dtype('uint16'), 'Labeled'), (2, 1, dtype('uint32'), 'Mask'), (2, 1, dtype('uint8'), 'Synthetic'), (2, 1, dtype('<u2'), 'Synthetic'), (2, 1, dtype('uint16'), 'Mask'), (3, 3, dtype('uint8'), 'Synthetic'), (2, 1, dtype('int32'), 'Mask'), (2, 1, dtype('uint8'), 'Labeled')}
/cellpose : 1726 files
= {(2, 1, dtype('uint16'), 'Mask'), (2, 1, dtype('uint32'), 'Labeled'), (3, 3, dtype('uint8'), 'Labeled')}
/omnipose : 1230 files
= {(2, 1, dtype('int8'), 'Mask'), (2, 1, dtype('uint16'), 'Mask'), (2, 1, dtype('uint16'), 'Labeled'), (2, 1, dtype('int16'), 'Mask')}
/livecell : 5848 files
= {(2, 1, dtype('uint8'), 'Labeled')}
/sciencebowl : 33215 files
= {(3, 4, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint8'), 'Mask'), (3, 3, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint16'), 'Labeled')}      

/neurips : 3158 files
= {(2, 1, dtype('uint16'), 'Mask'), (3, 3, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint8'), 'Mask'), (2, 1, dtype('uint8'), 'Synthetic'), (2, 1, dtype('uint32'), 'Mask'), (2, 1, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint16'), 'Synthetic')}
"""