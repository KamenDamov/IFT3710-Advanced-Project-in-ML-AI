import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile as tif
import pandas as pd
from skimage import io, color
import PIL.Image as Image
import numpy as np
import colorsys
import zipfile
import os
import cv2

from src.datasets.datasets import *

# Save black-white mask
def save_bw_mask(source, target):
    imgT = load_image(source)
    im = Image.fromarray((imgT != 0).astype('uint8')*255)
    save_image(target, im)

# Save grayscale mask
def save_gray_mask(source, target):
    imgT = load_image(source)
    num_objects = imgT.max()
    im = Image.fromarray((imgT / num_objects * 255).astype('uint8'))
    save_image(target, im)

# TOO SLOW
# Save hue-vector mask
def save_hue_mask(root, store, datapath, df):
    imgT = load_image(root + datapath)
    imgTC = np.zeros((imgT.shape[0], imgT.shape[1], 3), dtype=np.float32)
    for index, row in df[1:].iterrows():
        i = row['ID']
        bounds = [row['Top'], row['Bottom'], row['Left'], row['Right']]
        middle = np.array([row['Y'], row['X']])
        rows, cols = np.where(imgT[bounds[0]:bounds[1], bounds[2]:bounds[3]] == i)
        rows += bounds[0]
        cols += bounds[2]
        for r, c in zip(rows, cols):
            location = np.array([r, c])
            vector = (location - middle)/np.array([bounds[1] - bounds[0], bounds[3] - bounds[2]])
            # Get euclidian norm
            norm = np.linalg.norm(vector)
            angle = np.arctan2(vector[1], vector[0])  # Calculate the angle in radians
            hue = (angle + np.pi) / (2 * np.pi)  # Normalize angle to [0, 1] for hue
            rgb = colorsys.hsv_to_rgb(hue, norm, 1.0)  # Convert HSV to RGB
            imgTC[r, c] = rgb
    print(imgTC.min(), imgTC.max())
    im = Image.fromarray((imgTC * 255).astype('uint8'), mode="RGB")
    folder, name, ext = split_filepath(datapath)
    target = store + folder
    os.makedirs(target, exist_ok=True)
    maskfile = target + name + ".vect.png"
    im.save(maskfile)

def sanity_check(dataset, sample):
    image = dataset.load(sample.raw_image)
    #print(image.shape, sample.raw_image)
    if len(image.shape) == 3:
        assert image.shape[2] == 3
        channels = image.sum(axis=(0, 1))
        for channel in range(3):
            if channels[channel] == 0:
                print("Found empty channel: ", image.shape, channel, sample.raw_image)
    tensor = dataset.load(sample.raw_mask)
    dense = (len(np.unique(tensor)) == tensor.max()+1)
    if 65535 in tensor:
        print("Found 65535 in: ", sample.raw_mask)
    if not dense:
        print("Found non-dense mask: ", sample.raw_mask)
    df = DataLabels(sample.meta_frame).df
    width = df['Right'].max()
    height = df['Bottom'].max()
    assert tensor.shape[0] == height
    assert tensor.shape[1] == width
    area = df['Area'].sum()
    assert (width * height) == area
    assert area != 0
    for index in range(len(df)):
        bdf = df.iloc[index]
        sanity_check_box(width, height, bdf)

def sanity_check_box(width, height, bdf):
    bwidth = bdf['Right'] - bdf['Left']
    bheight = bdf['Bottom'] - bdf['Top']
    barea = bwidth * bheight
    #print(width, height, bdf)
    assert 0 <= bwidth <= width
    assert 0 <= bheight <= height
    assert 0 <= bdf['Area'] <= barea
    assert 0 <= bdf['Left'] <= width
    assert 0 <= bdf['Right'] <= width
    assert 0 <= bdf['Top'] <= height
    assert 0 <= bdf['Bottom'] <= height

def check_signatures(dataroot, datasets):
    for dataset in datasets:
        files = list(dataset.enumerate(dataroot))
        print(dataset.root, ":", len(files), "files")
        signatures = set(dataset.signature(dataroot + filepath) for filepath, cat in tqdm(files))
        print("=", signatures)

if False and __name__ == "__main__":
    for dataset in DataSet.filesets:
        dataset.unpack("./data/raw")

if __name__ == "__main__":
    dataset = DataSet("./data")
    for sample in tqdm(dataset, desc="Preparing metadata frames"):
        pass
        #sample.prepare_frame()
        #sanity_check(dataset, sample)
        # Also save human-readable mask images, for debugging
        #safely_process([], save_clean_image)(sample.raw_image, sample.clean_image)
        #safely_process([], save_bw_mask)(sample.raw_mask, sample.bw_mask)
        #safely_process([], save_gray_mask)(sample.raw_mask, sample.gray_mask)

"""
/neurips : 3158 files
= {(3, 3, dtype('uint8'), 'Labeled'), (2, 1, dtype('<u2'), 'Labeled'), (2, 1, dtype('int32'), 'Labeled'), (2, 1, dtype('float64'), 'Labeled'), (3, 3, dtype('uint16'), 'Labeled'), (2, 1, dtype('uint32'), 'Mask'), (2, 1, dtype('uint8'), 'Synthetic'), (2, 1, dtype('<u2'), 'Synthetic'), (2, 1, dtype('uint16'), 'Mask'), (3, 3, dtype('uint8'), 'Synthetic'), (2, 1, dtype('int32'), 'Mask'), (2, 1, dtype('uint8'), 'Labeled')}
/cellpose : 1726 files
= {(2, 1, dtype('uint16'), 'Mask'), (2, 1, dtype('uint32'), 'Labeled'), (3, 3, dtype('uint8'), 'Labeled')}
/omnipose : 1230 files
= {(2, 1, dtype('int8'), 'Mask'), (2, 1, dtype('uint16'), 'Mask'), (2, 1, dtype('uint16'), 'Labeled'), (2, 1, dtype('int16'), 'Mask')}
/livecell : 5848 files
= {(2, 1, dtype('uint8'), 'Labeled')}
/sciencebowl : 33215 files
= {(3, 4, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint8'), 'Mask'), (3, 3, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint16'), 'Labeled')}      

/neurips : 3158 files
= {(2, 1, dtype('uint16'), 'Mask'), (3, 3, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint8'), 'Mask'), (2, 1, dtype('uint8'), 'Synthetic'), (2, 1, dtype('uint32'), 'Mask'), (2, 1, dtype('uint8'), 'Labeled'), (2, 1, dtype('uint16'), 'Synthetic')}
"""