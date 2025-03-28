import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Pix2PixDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(256, 256)):
        """
        Args:
        - image_dir (str): Path to raw input images.
        - label_dir (str): Path to corresponding segmentation masks.
        - image_size (tuple): Target image size (default 256x256 for Pix2Pix).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = sorted(os.listdir(image_dir))  # Ensure matching order
        self.mask_filenames = sorted(os.listdir(label_dir))

        self.transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.label_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        return {'A': image, 'B': mask}  # Pix2Pix expects (input, target)

