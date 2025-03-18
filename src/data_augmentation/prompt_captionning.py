import torch
import os
import logging
import argparse
import json
from tqdm import tqdm
from typing import List, Dict
from promptcap import PromptCap

# Configure logging
logging.basicConfig(
    filename="logs.txt",
    filemode="w",  # Overwrite logs on each run
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_images_path(folder_path: str) -> List[str]:
    """Loads image paths from a given folder."""
    images = []
    
    try:
        with os.scandir(folder_path) as entries:
            for entry in tqdm(entries, desc="Loading image paths"):
                if entry.is_file():
                    images.append(entry.path)
    except Exception as e:
        logging.error(f"Error loading images from {folder_path}: {e}")

    return images

def get_images_captions(images_path: List[str], prompt: str, model: PromptCap) -> List[Dict[str, str]]:
    """Generates captions for a list of image paths using a given model."""
    results = []

    for image in tqdm(images_path, desc="Generating Captions"):
        try:
            caption = model.caption(prompt, image)
            results.append({"image": image, "caption": caption})
        except Exception as e:
            logging.error(f"Failed to generate caption for {image}: {e}")

    return results

def save_captions(data: List[Dict[str, str]], output_path: str, file_name: str = "captions.json") -> None:
    """Saves generated captions to a JSON file."""
    os.makedirs(output_path, exist_ok=True)

    try:
        with open(os.path.join(output_path, file_name), "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        logging.error(f"Error saving captions to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prompt Captioning for Images")
    parser.add_argument("--images_folder", default="../../data/preprocessing_outputs/transformed_images_labels/images", type=str, help="Path to folder containing images")
    parser.add_argument("--prompt", default="how many cells are in this image?", type=str, help="Prompt for caption generation")
    parser.add_argument("--output_file", default="../../data/preprocessing_outputs/captions", type=str, help="Output folder for captions JSON")
    
    args = parser.parse_args()

    images = load_images_path(args.images_folder)

    if not images:
        print("No valid images found. Exiting.")
        return

    # Load model once
    model = PromptCap("tifa-benchmark/promptcap-coco-vqa")
    
    if torch.cuda.is_available():
        model.cuda()

    images_captions = get_images_captions(images, args.prompt, model)
    save_captions(images_captions, args.output_file)

if __name__ == "__main__":
    main()
