import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, T5ForConditionalGeneration, T5Tokenizer
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer, util
import argparse

# Load models
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

blip_processor = BlipProcessor.from_pretrained("../models/blip")
blip_model = BlipForConditionalGeneration.from_pretrained("../models/blip")

clip_processor = CLIPProcessor.from_pretrained("../models/clip")
clip_model = CLIPModel.from_pretrained("../models/clip")

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

t5_tokenizer = T5Tokenizer.from_pretrained("../models/t5")
t5_model = T5ForConditionalGeneration.from_pretrained("../models/t5")

# Function to generate captions
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption = blip_model.generate(**inputs)
    return blip_processor.batch_decode(caption, skip_special_tokens=True)[0]

# Function to filter captions using CLIP similarity
def filter_caption(image_path, caption):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity_score = outputs.logits_per_text.item()
    return caption if similarity_score > 0.5 else None  # Threshold tuning required

# Function to augment captions using T5 paraphrasing
def augment_caption(caption):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    input_text = f"paraphrase: {caption}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    image_path = "../../data/preprocessing_outputs/normalized_data/images/cell_00055.png"
    # prompt = "You will receive an image of cells and describe the type and the number of cells in the image."
    generated_caption = generate_caption(image_path)
    filtered_caption = filter_caption(image_path, generated_caption)
    if filtered_caption:
        augmented_caption = augment_caption(filtered_caption)
        print("Original Caption:", generated_caption)
        print("Filtered Caption:", filtered_caption)
        print("Augmented Caption:", augmented_caption)
    else:
        print("Caption discarded due to low relevance.")

