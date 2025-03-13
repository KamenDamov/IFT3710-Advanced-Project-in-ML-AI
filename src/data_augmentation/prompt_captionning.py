import torch
import os
from promptcap import PromptCap
import argparse
import json
from tqdm import tqdm

def load_images_path(folder_path):
    images = []
    logs = []
    for _, image in enumerate(tqdm(os.listdir(folder_path), desc="Loading image path")):
        try:
            image_path = os.path.join(folder_path, image)
            images.append(image_path)
        except Exception as e:
            logs.append(image)
    if logs:
        print("Some images could not be loaded. Check logs.txt for more information.")
        with open('logs.txt', 'a') as f: 
           f.write("\n".join(logs))
           f.close()   
    return images   

def get_images_captions(images_path, prompt):
    result = []

    model = PromptCap("tifa-benchmark/promptcap-coco-vqa")

    if torch.cuda.is_available():
        model.cuda()

    for _, image in enumerate(tqdm(images_path, desc="Generating Captions")):
        caption = model.caption(prompt, image)
        result.append({"image": image, "caption": caption})
    
    return result

def save_captions(data, output_path, file_name="captions.json"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, file_name), "w") as file:
        json.dump(data, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt captionning")
    parser.add_argument("--images_folder", default="../../data/preprocessing_outputs/transformed_images_labels/images", type=str, help="Images path")
    parser.add_argument("--prompt", default="how many cells are in this image?", type=str, help="Prompt")
    parser.add_argument("--output_file", default="../../data/preprocessing_outputs/captions", type=str, help="Output file")
    args = parser.parse_args()
    images = load_images_path(args.images_folder)
    images_captions = get_images_captions(images, args.prompt)
    save_captions(images_captions, args.output_file)
