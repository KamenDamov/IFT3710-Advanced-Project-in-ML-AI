from src.manage_files import *

###
# Download the DataScienceBowl dataset from:
# https://www.kaggle.com/competitions/data-science-bowl-2018/data
#
# Raw file structure:
# /sciencebowl
# ├── data-science-bowl-2018.zip            ***
# ├── stage1_sample_submission.csv
# ├── stage1_solution.csv
# ├── stage1_test.zip
# ├── stage1_train.zip
# ├── stage1_train_labels.csv
# ├── stage2_sample_submission_final.csv
# ├── stage2_test_final.zip
# ├── /stage1_test
# ├── /stage1_train
# └── /stage2_test_final
###
class ScienceBowlSet(BaseFileSet):
    def __init__(self):
        super().__init__("/sciencebowl")

    def unpack(self, dataroot):
        needs_offset = ["stage1_train", "stage1_test", "stage2_test_final"]
        unzip_dataset(dataroot, self.root + "/", needs_offset)
        #build_masks(self, dataroot)

    def mask_filepath(self, filepath):
        if "/images" in filepath:
            folder, name, ext = split_filepath(filepath)
            target = folder + name + "_label.tiff"
            yield (target, MASK)
    
    def categorize(self, filepath):
        if "/masks" in filepath:
            return MASK
        if "_label.tiff" in filepath:
            return MASK
        if "stage1_train" in filepath and "/images" in filepath:
            return LABELED
        return UNLABELED

def build_masks(dataset, dataroot):
    for filepath in dataset.enumerate(dataroot):
        if dataset.categorize(filepath) == LABELED:
            folder, name, ext = split_filepath(dataroot + filepath)
            (target, _), = dataset.mask_filepath(dataroot + filepath)
            maskfolder = folder.replace("/images", "/masks")
            if os.path.exists(maskfolder):
                safely_process([], mask_builder(folder))(filepath, target)

def mask_builder(folder):
    def build(filepath, target):
        mask_paths = [folder + mask for mask in os.listdir(folder)]
        # Stack masks with unique labels
        combined_mask = np.zeros_like(io.imread(mask_paths[0]), dtype=np.uint16)
        for label, mask_path in enumerate(mask_paths, start=1):
            mask = io.imread(mask_path)
            combined_mask[mask > 0] = label  # Assign unique label
        save_image(target, combined_mask)
    return build

def rle_decode(mask_rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint16)
    if pd.isna(mask_rle):
        return mask.reshape(shape)

    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    for i, (start, length) in enumerate(zip(starts, lengths), start=1):
        mask[start: start + length] = i

    return mask.reshape(shape)

if __name__ == '__main__':
    root = "./data/raw"
    dataset = ScienceBowlSet()
    dataset.unpack(root)
