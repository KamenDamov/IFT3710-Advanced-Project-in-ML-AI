from src.manage_files import *

###
# Download the DataScienceBowl dataset from:
# https://www.kaggle.com/competitions/data-science-bowl-2018/data
#
# Raw file structure:
# /sciencebowl
# └── data-science-bowl-2018.zip
###
class ScienceBowlSet(BaseFileSet):
    def __init__(self):
        super().__init__("/sciencebowl")

    def mask_filepath(self, filepath):
        if "/images" in filepath:
            yield (filepath.replace("/images", "/labels"), MASK)
    
    def categorize(self, filepath):
        if "/masks" in filepath:
            return MASK
        if "/images" in filepath:
            return LABELED
        return UNLABELED

if __name__ == '__main__':
    root = "./data/raw"
    dataset = ScienceBowlSet()
    dataset.unpack(root)
