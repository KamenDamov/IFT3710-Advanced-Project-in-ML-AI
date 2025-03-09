import os
import numpy as np
from cellpose import models, io, train
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Paths
TRAIN_DIR_X_TRAIN = "data/preprocessing_outputs/transformed_images_labels/images"  # Folder with training images & masks
TRAIN_DIR_Y_TRAIN = "data/preprocessing_outputs/transformed_images_labels/labels"
TRAIN_DIR_X_TEST = "data/preprocessing_outputs/tuning/transformed_images_labels/images"  # Folder with training images & masks
TRAIN_DIR_Y_TEST = "data/preprocessing_outputs/tuning/transformed_images_labels/labels"
MODEL_SAVE_PATH = "src/models/cellpose3"  # Where to save the trained model
TEST_IMG_PATH = "data/preprocessing_outputs/tuning/transformed_images_labels/images/cell_00001.png"  # A test image for evaluation
OUTPUT_MASK_PATH = ""  # Where to save segmentation mask

# Training parameters
PRETRAINED_MODEL = "nuclei"  # Can be "cyto", "nuclei", or None for training from scratch
N_EPOCHS = 5  # Training epochs
USE_GPU = True  # Set to False if no GPU is available

# Choose input mode: 'grayscale', 'rgb_single', or 'rgb_dual'
INPUT_MODE = "rgb_dual"  # Options: "grayscale", "rgb_single", "rgb_dual"

# Set Channels based on Mode
if INPUT_MODE == "grayscale":
    CHANNELS = [0, 0]  # Convert RGB to grayscale
elif INPUT_MODE == "rgb_single":
    CHANNELS = [1, 0]  # Uses Green only (change index for different channels)
elif INPUT_MODE == "rgb_dual":
    CHANNELS = [1, 2]  # Uses Green & Blue channels

def load_images_from_folder(folder):
    image_list = []
    
    # Get all image files in directory
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image
            if img is not None:
                image_list.append(img)
    
    if not image_list:
        raise ValueError(f"No valid images found in {folder}.")

    return np.array(image_list)

def convert_img_npy(image_path, use_pil=False): 
    if use_pil:
        img = Image.open(image_path)  # Load using PIL
        img = np.array(img)  # Convert to NumPy array
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load using OpenCV
    
    # Ensure correct format (H, W) for grayscale or (H, W, C) for RGB
    if img.ndim == 3 and img.shape[2] == 4:  # Handle RGBA images
        img = img[:, :, :3]  # Remove Alpha channel (Convert RGBA → RGB)
    
    return img

def preprocess_image(image):
    """Convert (3, H, W) -> (H, W, 3) for Cellpose compatibility."""
    if image.ndim == 3 and image.shape[0] == 3:  # Check for (3, H, W) format
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)
    return image

def train_model():
    print("Loading training images and masks...")
    
    # Load images & masks from directory
    train_imgs = load_images_from_folder(TRAIN_DIR_X_TRAIN)

    # Convert from (3, H, W) to (H, W, 3) if needed
    train_imgs = np.array([preprocess_image(img) for img in train_imgs])

    test_imgs = load_images_from_folder(TRAIN_DIR_X_TEST)

    test_imgs = np.array([preprocess_image(img) for img in test_imgs])

    # Convert RGB to grayscale if required
    if INPUT_MODE == "grayscale":
        print("Converting RGB images to grayscale for training...")
        train_imgs = np.mean(train_imgs, axis=-1)  # Convert RGB to grayscale

    # Verify masks exist
    mask_files_train = load_images_from_folder(TRAIN_DIR_Y_TRAIN)#[f for f in os.listdir(TRAIN_DIR_Y) if f.endswith("_masks.npy")]
    mask_files_test = load_images_from_folder(TRAIN_DIR_Y_TEST)
    #if not mask_files:
    #   raise ValueError("No masks found! Ensure your masks are saved as '_masks.npy'.")

    print(f"Found {len(train_imgs)} training images with corresponding masks.")

    # Train model
    model = models.Cellpose(gpu=True, model_type='nuclei')
    diameters = []
    for img, mask in zip(train_imgs, mask_files_train):
        _, _, _, diameter = model.eval(img, mask, channels=CHANNELS)
        diameters.append(diameter)

    diameters = np.array(diameters)
    if len(train_imgs) == 0 or len(mask_files_train) == 0:
        raise ValueError("Error: No training data available after filtering. Check your masks!")

    # Train model
    train.train_seg(
        model,
        train_data=train_imgs,
        train_labels=mask_files_train,  # Load masks
        test_data=test_imgs,
        test_labels=mask_files_test,
        diam_labels=diameters,
        normalize=True,
        SGD=True,
        weight_decay=1e-4,
        learning_rate=0.1,
        n_epochs=N_EPOCHS,
        save_path=MODEL_SAVE_PATH,
        channels=CHANNELS,
        min_train_masks=1
    )
    
    print(f"Model training completed. Model saved at {MODEL_SAVE_PATH}")

def evaluate_model():
    print("Loading trained model for evaluation...")
    
    # Load trained model
    custom_model = models.CellposeModel(pretrained_model=MODEL_SAVE_PATH, gpu=USE_GPU)

    # Check if test image exists
    if not os.path.exists(TEST_IMG_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMG_PATH}")

    # Load test image
    test_img = io.imread(TEST_IMG_PATH)

    # Convert (3, H, W) to (H, W, 3) if necessary
    test_img = preprocess_image(test_img)

    # Convert to grayscale if necessary
    if INPUT_MODE == "grayscale":
        print("Converting test RGB image to grayscale for evaluation...")
        test_img = np.mean(test_img, axis=-1)  # Convert RGB to grayscale

    # Run segmentation
    print("Running Cellpose segmentation...")
    masks, flows, styles, diams = custom_model.eval(test_img, diameter=None, channels=CHANNELS)

    # Save mask
    io.imsave(OUTPUT_MASK_PATH, masks.astype(np.uint16))
    print(f"Segmentation mask saved at {OUTPUT_MASK_PATH}")

    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(test_img, cmap="gray" if test_img.ndim == 2 else None)
    ax[0].set_title("Test Image")
    ax[1].imshow(masks, cmap="jet")
    ax[1].set_title("Segmentation Mask")
    plt.show()

if __name__ == "__main__":
    train_model()  # Train model
    evaluate_model()  # Evaluate on test image
