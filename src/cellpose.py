from src.manage_files import *

###
# Download the CellPose dataset from:
# https://www.cellpose.org/dataset
# 
# Raw file structure:
# /cellpose
# ├── test.zip              ***
# ├── train.zip             ***
# ├── train_cyto2.zip       ***
# ├── /test
# ├── /train
# └── /train_cyto2
###
class CellposeSet(BaseFileSet):
    def __init__(self):
        super().__init__("/cellpose")

    def mask_filepath(self, filepath):
        if ("img" in filepath):
            yield filepath.replace("img", "masks"), MASK
    
    def categorize(self, filepath):
        if "masks" in filepath:
            return MASK
        elif "img" in filepath:
            return IMAGE
    
    def blacklist(self, filepath):
        # This mask is corrupted
        return "/train_cyto2/758" in filepath
    
    def load(self, filepath):
        category = self.categorize(filepath)
        if category == IMAGE:
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

if __name__ == '__main__':
    root = "./data/raw"
    dataset = CellposeSet()
    dataset.unpack(root)
    dataset.crosscheck(root)
