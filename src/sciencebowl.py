import pandas as pd

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
        build_masks(self, dataroot)

    def mask_filepath(self, filepath):
        if "/images" in filepath:
            folder, name, ext = split_filepath(filepath)
            target = folder.replace(f"{name}/images", "labels") + name + ".tiff"
            yield (target, MASK)
    
    def categorize(self, filepath):
        if "/labels" in filepath:
            return MASK
        if "/images" in filepath:
            return LABELED
        if "/masks" in filepath:
            return MASK
        return UNLABELED

def convert_test_from_csv(csv_path, destination):
    for image_id, masks in pd.read_csv(csv_path).groupby('ImageId'):
        source = destination + f"/{image_id}/images/{image_id}.png"
        target = destination + f"/labels/{image_id}.tiff"
        safely_process([], mask_builder_from_csv(masks))(source, target)

def mask_builder_from_csv(masks):
    def build(filepath, target):
        # Reset mask for each image
        labeled_mask = None
        for idx, (_, row) in enumerate(masks.iterrows(), start=1):
            encoded_pixels, height, width = row['EncodedPixels'], row['Height'], row['Width']
            shape = (int(height), int(width))

            labeled_mask = np.zeros(shape, dtype=np.uint16) if (labeled_mask is None) else labeled_mask
            # Decode RLE mask
            mask = rle_decode(encoded_pixels, shape)
            # Label connected components to assign unique IDs
            labeled_mask[mask] = idx
        if labeled_mask.max() < 256:
            # Convert to uint8 if less than 256 labels
            labeled_mask = labeled_mask.astype(np.uint8)
        # Save mask as TIFF
        save_image(target, labeled_mask)
    return build

def rle_decode(mask_rle, shape):
    shape = shape[1], shape[0]  # (width, height)
    mask = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    if pd.isna(mask_rle):
        return mask.reshape(shape)
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    for (start, length) in zip(starts, lengths):
        mask[start: start + length] = True
    return mask.reshape(shape).transpose()

def build_masks(dataset, dataroot):
    #convert_test_from_csv(dataroot + dataset.root + "/stage1_train_labels.csv", dataroot + dataset.root + "/stage1_train")
    convert_test_from_csv(dataroot + dataset.root + "/stage1_solution.csv", dataroot + dataset.root + "/stage1_test")
    for filepath in dataset.enumerate(dataroot):
        if dataset.categorize(filepath) == LABELED:
            if "stage1_train" in filepath:
                folder, name, ext = split_filepath(dataroot + filepath)
                target = folder.replace(f"{name}/images", "labels") + name + ".tiff"
                maskfolder = folder.replace("/images", "/masks")
                safely_process([], mask_builder(maskfolder))(filepath, target)

def mask_builder(maskfolder):
    def build(filepath, target):
        mask_paths = [maskfolder + mask for mask in os.listdir(maskfolder)]
        # Stack masks with unique labels
        combined_mask = np.zeros_like(load_image(mask_paths[0]), dtype=np.uint16)
        for label, mask_path in enumerate(mask_paths, start=1):
            mask = load_image(mask_path)
            combined_mask[mask > 0] = label  # Assign unique label
        save_image(target, combined_mask)
    return build

if __name__ == '__main__':
    root = "./data/raw"
    dataset = ScienceBowlSet()
    dataset.unpack(root)
