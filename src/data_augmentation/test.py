import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

torch.cuda.empty_cache()

# Load BLIP-2 model & processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    caption = generate_caption("../../data/preprocessing_outputs/transformed_images_labels/images/cell_00001.png")
    print("Caption:", caption)
