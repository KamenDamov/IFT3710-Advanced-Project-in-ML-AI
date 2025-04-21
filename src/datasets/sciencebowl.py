import pandas as pd

from .manage_files import *

###
# Download the DataScienceBowl dataset from:
# https://www.kaggle.com/competitions/data-science-bowl-2018/data
# https://bbbc.broadinstitute.org/BBBC038/
#
# Raw file structure:
# /sciencebowl
# ├── stage2_solution_final.csv            ***
# ├── data-science-bowl-2018.zip           ***
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

    def blacklist(self, filepath):
        # Don't process partial masks
        return "/masks" in filepath
    
    def mask_filepath(self, filepath):
        if "/images" in filepath:
            folder, name, ext = split_filepath(filepath)
            target = folder.replace(f"{name}/images", "labels") + name + ".tiff"
            yield (target, MASK)
    
    def categorize(self, filepath):
        if "/labels" in filepath:
            return MASK
        if "/images" in filepath:
            return IMAGE

def convert_test_from_csv(csv_path, destination):
    df = pd.read_csv(csv_path)
    groups = list(df.groupby('ImageId'))
    for image_id, masks in tqdm(groups, desc="Processing labels " + csv_path):
        source, target = sample_paths(destination, image_id)
        if ignore_sample(masks):
            #os.remove(source)
            continue
        safely_process([], mask_builder_from_csv(masks))(source, target)

def ignore_sample(masks):
    usage = masks.iloc[0]['Usage']
    return usage not in ['Public', 'Private'] # == 'Ignored'

def sample_paths(destination, image_id):
    source = destination + f"/{image_id}/images/{image_id}.png"
    target = destination + f"/labels/{image_id}.tiff"
    return source, target

def mask_builder_from_csv(masks):
    def build(filepath, target):
        # Reset mask for each image
        labeled_mask = None
        for idx, (_, row) in enumerate(masks.iterrows(), start=1):
            encoded_pixels, height, width = row['EncodedPixels'], row['Height'], row['Width']
            shape = (int(height), int(width))
            # Initialize mask if not already done
            labeled_mask = np.zeros(shape, dtype=np.uint16) if (labeled_mask is None) else labeled_mask
            # Decode RLE mask
            mask = rle_decode(encoded_pixels, shape)
            # Label connected components to assign unique IDs
            labeled_mask[mask] = idx
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
    convert_test_from_csv(dataroot + dataset.root + "/stage1_solution.csv", dataroot + dataset.root + "/stage1_test")
    convert_test_from_csv(dataroot + dataset.root + "/stage2_solution_final.csv", dataroot + dataset.root + "/stage2_test_final")
    train_files = [filepath for filepath, cat in dataset.enumerate(dataroot) if cat == IMAGE and "/stage1_train" in filepath]
    for filepath in tqdm(train_files, desc="Processing mask labels"):
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
    dataset.crosscheck(root)
