
from .manage_files import *
from .neurips import ZenodoNeurIPS
from .cellpose import CellposeSet
from .omnipose import OmniPoseSet
from .livecell import LiveCellSet
from .sciencebowl import ScienceBowlSet

### Dataset sources ###
# MEDIAR
# https://github.com/Lee-Gihun/MEDIAR
###

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

def maskframe_builder(dataset):
    def save_maskframe(mask_path, frame_path):
        tensor = dataset.load(mask_path)
        maskframe = mask_frame(tensor)
        maskframe.to_csv(frame_path)
    return save_maskframe

def prepare_metaframe(dataset, target_path):
    rawroot = dataset.dataroot + "/raw"
    data_map = dataset_frame(rawroot, dataset.filesets)
    data_map.to_csv(target_path)

def dataset_frame(root, matchers):
    numbers = []
    for matcher in matchers:
        matcher.unpack(root)
        matcher.crosscheck(root)
    for matcher in matchers:
        numbers += collect_dataset(root, matcher)
    frame = pd.DataFrame(numbers, columns = ["Path", "Mask", "Width", "Height", "Objects", "Background", "Synthetic"])
    frame = frame.set_index("Path")
    return frame.sort_index()

def collect_dataset(root, matcher):
    files = list(matcher.enumerate(root))
    for imagepath, category in tqdm(files, desc="Compiling dataset csv " + matcher.root):
        if category == IMAGE:
            for maskpath, category in matcher.mask_filepath(imagepath):
                imgT = matcher.load(root + imagepath)
                yield build_metarow(imagepath, maskpath, category, imgT)

def build_metarow(imagepath, maskpath, category, imgT):
    # Get global statistics
    background = (imgT == 0).sum()
    num_objects = imgT.max()
    synthetic = (category == SYNTHETIC)
    return {"Path":imagepath, "Mask":maskpath, "Synthetic": synthetic, "Width": imgT.shape[1], "Height":imgT.shape[0], "Objects": num_objects, "Background": background}

class DataSet:
    filesets = [ZenodoNeurIPS(), CellposeSet(), OmniPoseSet(), LiveCellSet(), ScienceBowlSet()]

    def __init__(self, dataroot, filesets=None):
        self.dataroot = dataroot
        self.filesets = filesets or DataSet.filesets
        self.meta_frame = self.dataroot + "/dataset.labels.csv"
        self.df = self.prepare_frame()

    def __str__(self):
        return self.dataroot
    
    def __len__(self):
        return len(self.df)
    
    def prepare_frame(self):
        safely_process([], prepare_metaframe)(self, self.meta_frame)
        return pd.read_csv(self.meta_frame)

    def __iter__(self):
        for index in range(len(self.df)):
            row = self.df.iloc[index]
            sample = DataSample(self, row)
            if not self.blacklist(sample):
                yield sample

    def blacklist(self, sample):
        return sample.df["Synthetic"] \
            or ("WSI" in sample.df["Path"]) # Large images
    
    # Load a well-formed tensor from the raw image
    def load(self, filepath):
        if '/raw/' not in filepath:
            return load_image(filepath)
        for dataset in self.filesets:
            if dataset.root in filepath:
                return dataset.load(filepath)

class DataSample:
    def __init__(self, dataset, df):
        self.dataset = dataset
        self.dataroot = dataset.dataroot
        self.df = df

        self.name = split_filepath(df["Path"])[1]
        self.init_paths(df["Path"], df["Mask"])
        self.width = df.get('Width')
        self.height = df.get('Height')
    
    def __str__(self):
        return str(self.df)
    
    def prepare_frame(self):
        safely_process([], maskframe_builder(self.dataset))(self.raw_mask, self.meta_frame)

    def labels(self):
        self.prepare_frame(self)
        return DataLabels(self.meta_frame)
    
    def init_paths(self, image_path, mask_path):
        self.raw_image = self.dataroot + "/raw" + image_path
        self.raw_mask = self.dataroot + "/raw" + mask_path
        # WARN: These paths could conflict if we enable synthetic masks
        self.image = self.dataroot + "/unify" + target_file(image_path, ".png")
        self.mask = self.dataroot + "/unify" + target_file(image_path, ".tiff")
        self.meta_frame = self.dataroot + "/unify" + target_file(image_path, ".csv")
        self.clean_image = self.dataroot + "/processed" + target_file(image_path, ".png")
        self.bw_mask = self.dataroot + "/processed" + target_file(mask_path, ".bin.png")
        self.gray_mask = self.dataroot + "/processed" + target_file(mask_path, ".gray.png")
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

if __name__ == "__main__":
    for sample in tqdm(DataSet("./data"), desc="Compiling segmentation csv frames"):
        sample.prepare_frame()
