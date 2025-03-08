import os
import numpy as np
from cellpose import models, io
import matplotlib.pyplot as plt

# Paths
TRAIN_DIR = "path/to/dataset"  # Folder with training images & masks
MODEL_SAVE_PATH = "path/to/save/model"  # Where to save the trained model
TEST_IMG_PATH = "path/to/test/image.png"  # A test image for evaluation
OUTPUT_MASK_PATH = "path/to/output_mask.png"  # Where to save segmentation mask

# Training parameters
PRETRAINED_MODEL = "nuclei"  # Can be "cyto", "nuclei", or None for training from scratch
N_EPOCHS = 500  # Training epochs
CHANNELS = [0, 0]  # Channels: [grayscale, second channel (0 if not applicable)]
USE_GPU = True  # Set to False if no GPU is available
AUGMENTATION = True  # Enable augmentations

def train_model():
    # Load images & masks from directory
    train_imgs = io.imread(TRAIN_DIR)  # Automatically finds _masks.npy if present

    # Initialize Cellpose model
    model = models.CellposeModel(pretrained_model=PRETRAINED_MODEL, gpu=USE_GPU)

    # Train model
    model.train(
        train_imgs,
        train_masks=None,  # Assumes masks are stored with _masks.npy suffix
        n_epochs=N_EPOCHS,
        save_path=MODEL_SAVE_PATH,
        channels=CHANNELS,
        augment=AUGMENTATION,
    )
    print(f"Model saved at {MODEL_SAVE_PATH}")

def evaluate_model():
    # Load trained model
    custom_model = models.CellposeModel(pretrained_model=MODEL_SAVE_PATH, gpu=USE_GPU)

    # Load test image
    test_img = io.imread(TEST_IMG_PATH)

    # Run segmentation
    masks, flows, styles, diams = custom_model.eval(test_img, diameter=None, channels=CHANNELS)

    # Save mask
    io.imsave(OUTPUT_MASK_PATH, masks.astype(np.uint16))
    print(f"Segmentation mask saved at {OUTPUT_MASK_PATH}")

    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(test_img, cmap="gray")
    ax[0].set_title("Test Image")
    ax[1].imshow(masks, cmap="jet")
    ax[1].set_title("Segmentation Mask")
    plt.show()

if __name__ == "__main__":
    train_model()  # Train model
    evaluate_model()  # Evaluate on test image
