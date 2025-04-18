from src.manage_files import *

###
# Download the OmniPose dataset from:
# https://github.com/kevinjohncutler/omnipose
# https://www.nature.com/articles/s41592-022-01639-4#data-availability
# https://osf.io/xmury/files/osfstorage
# 
# Raw file structure:
# /omnipose
# ├── datasets.zip         ***
# ├── /bact_fluor
# |   ├── /test_sorted
# |   └── /train_sorted
# ├── /bact_phase
# |   ├── /test_sorted
# |   └── /train_sorted
# ├── /worm
# └── /worm_high_res
###
class OmniPoseSet(BaseFileSet):
    def __init__(self):
        super().__init__("/omnipose")

    def blacklist(self, filepath):
        return ("/worm" in filepath) \
            or ("_flows" in filepath)
    
    def mask_filepath(self, filepath):
        folder, name, ext = split_filepath(filepath)
        yield (folder + name + "_masks" + ext), MASK
    
    def categorize(self, filepath):
        if "_masks" in filepath:
            return MASK
        return IMAGE

if __name__ == '__main__':
    root = "./data/raw"
    dataset = OmniPoseSet()
    dataset.unpack(root)
    dataset.crosscheck(root)
