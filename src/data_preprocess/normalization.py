from skimage import io, segmentation, morphology, exposure
import tifffile as tif
from tqdm import tqdm
import numpy as np
import argparse
import os
join = os.path.join

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def to_single_channel_inst_map(label):
    """
    Convert label image to a single-channel integer map.
    If 'label' is already single-channel (H,W), return as is.
    If 'label' is 3-channel (H,W,3), convert each unique color to a unique integer.

    Parameters
    ----------
    label : np.ndarray
        Shape can be (H,W) or (H,W,3).

    Returns
    -------
    inst_map : np.ndarray, shape (H,W)
        Single-channel label image with integer IDs (0 for background).
    """
    
    # If single-channel, assume it's already integer-labeled
    if label.ndim == 2:
        # Make sure it’s an integer type. Otherwise, convert.
        if not np.issubdtype(label.dtype, np.integer):
            return label.astype(np.int32)
        return label
    
    # If 3-channel, convert from RGB to integer IDs
    elif label.ndim == 3 and label.shape[2] == 3:
        H, W, _ = label.shape
        label_reshaped = label.reshape(-1, 3)
        
        # Get unique colors
        unique_colors = np.unique(label_reshaped, axis=0)
        
        # Map each unique color to a unique ID
        color2id = {}
        current_id = 1
        
        for color in unique_colors:
            ctuple = tuple(color)
            # Assume (0,0,0) is background
            if ctuple == (0,0,0):
                color2id[ctuple] = 0
            else:
                color2id[ctuple] = current_id
                current_id += 1
        
        # Build output inst_map
        inst_map = np.zeros((H * W,), dtype=np.int32)
        
        for ctuple, id_val in color2id.items():
            mask = np.all(label_reshaped == ctuple, axis=1)
            inst_map[mask] = id_val
        
        return inst_map.reshape(H, W)
    
    else:
        raise ValueError(
            f"Unsupported label shape {label.shape}. "
            "Expected (H,W) or (H,W,3)."
        )

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
    print(boundary)
    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    print(interior_temp)
    print(boundary)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

def normalization(source_path, target_path):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Get images names
    images = source_path + "/images" #join(source_path, 'images')
    labels = source_path + "/labels" #join(source_path, 'labels')

    img_names = sorted(os.listdir(images))
    gt_names = [img_name.split('.')[0]+'_label.tiff' if img_name.split('.')[-1] == "tiff" else img_name.split('.')[0]+'.png' for img_name in img_names]

    # Create directories for preprocessed images and ground truth
    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)

    log = ["Failed to process images: \n"]
    for img_name, gt_name in zip(tqdm(img_names, desc="Normalizing images"), gt_names):

        try:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(images, img_name))
            else:
                img_data = io.imread(join(images, img_name))
            
            if gt_name.endswith('.tif') or gt_name.endswith('.tiff'):
                gt_data = tif.imread(join(labels, gt_name))
            else: 
                gt_data = io.imread(join(labels, gt_name))

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
            
            # conver instance bask to three-class mask: interior, boundary
            interior_map = create_interior_map(gt_data.astype(np.int16))
            
            io.imsave(join(target_path, 'images', img_name.split('.')[0]+'.png'), pre_img_data.astype(np.uint8), check_contrast=False)
            io.imsave(join(target_path, 'labels', gt_name.split('.')[0]+'.png'), interior_map.astype(np.uint8), check_contrast=False)
        except Exception as e: 
            print(e)
            log.append(img_name)      

    with open('logs.txt', 'a') as f: 
        f.write("\n".join(log))
        f.close()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applying Normalization")
    parser.add_argument("--source_path", default="../../data/data_science_bowl" , type=str, required=False, help="Path to input images.")
    parser.add_argument("--target_path", default="../../data/preprocessing_outputs/data_science_bowl/normalized_data" , type=str, required=False, help="Path to save transformed images.")
    args = parser.parse_args()
    normalization(args.source_path, args.target_path)