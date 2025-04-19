from .manage_files import *

###
# Download the NeurIPS Competition dataset from:
# https://zenodo.org/records/10719375
# 
# Raw file structure:
# /neurips
# ├── Testing.zip                   ***
# ├── Training-labeled.zip          ***
# ├── train-unlabeled-part1.zip     ***
# ├── train-unlabeled-part2.zip     ***
# ├── Tuning.zip                    ***
# ├── ReadMe.md                 
# ├── /release-part1
# ├── /Testing
# ├── /Training-labeled
# |   ├── /images
# |   └── /labels
# ├── /train-unlabeled-part2
# └── /Tuning
###

class ZenodoNeurIPS(BaseFileSet):
    def __init__(self):
        super().__init__("/neurips")

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
                    return IMAGE

if __name__ == '__main__':
    root = "./data/raw"
    dataset = ZenodoNeurIPS()
    dataset.unpack(root)
    dataset.crosscheck(root)
