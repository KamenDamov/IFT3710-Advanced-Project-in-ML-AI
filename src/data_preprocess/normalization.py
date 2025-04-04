from skimage import io, segmentation, morphology, exposure
import tifffile as tif
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os
from ..data_exploration import explore

join = os.path.join

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    #inst_map = to_single_channel_inst_map(inst_map)
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

def normalize_mask(mask_path, target_path):
    gt_data = load_image(mask_path)
    # conver instance bask to three-class mask: interior, boundary
    interior_map = create_interior_map(gt_data.astype(np.int16))
    io.imsave(target_path, interior_map.astype(np.uint8), check_contrast=False)

def normalize_image(img_path, target_path):
    img_data = load_image(img_path)

    # normalize image data
    if len(img_data.shape) == 2:
        img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        img_data = img_data[:,:, :3]
    else:
        pass
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:,:,i]
        if len(img_channel_i[np.nonzero(img_channel_i)])>0:
            pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
    
    io.imsave(target_path, pre_img_data.astype(np.uint8), check_contrast=False)

def load_image(img_path):
    dirpath, name, ext = explore.split_filepath(img_path)
    if ext in ['.tif', '.tiff']:
        return tif.imread(img_path)
    else:
        return io.imread(img_path)

def target_file(filepath, ext):
    dirpath, name, _ = explore.split_filepath(filepath)
    os.makedirs(dirpath, exist_ok=True)
    return dirpath + name + ext

def batch_process(log, source_path, target_path, extract, process):
    for name, df in explore.enumerate_frames(dataroot):
        if ".labels" in name:
            for filepath in tqdm(extract(df), desc="Normalizing images"):
                source_filepath = source_path + filepath
                target_filepath = target_file(target_path + filepath, ".png")
                if not os.path.exists(target_filepath):
                    try:
                        process(source_filepath, target_filepath)
                    except Exception as e: 
                        print(e)
                        log.append(source_filepath + " -> " + target_filepath)

def main(dataroot):
    norm_target = f"{dataroot}/preprocessing_outputs/normalized_data"
    log = ["Failed to process images: \n"]
    batch_process(log, f"{dataroot}/raw", norm_target, lambda df: df["Path"], normalize_image)
    batch_process(log, f"{dataroot}/raw", norm_target, lambda df: df["Mask"], normalize_mask)
            
    with open('logs.txt', 'a') as f:
        f.write("\n".join(log))
        f.close()

if __name__ == "__main__":
    dataroot = "../../data"
    parser = argparse.ArgumentParser(description="Applying Normalization")
    parser.add_argument("--source_path", default=f"{dataroot}/Training-labeled" , type=str, required=False, help="Path to input images.")
    parser.add_argument("--target_path", default=f"{dataroot}/preprocessing_outputs/normalized_data" , type=str, required=False, help="Path to save transformed images.")

    args = parser.parse_args()
    normalization(args.source_path, args.target_path)