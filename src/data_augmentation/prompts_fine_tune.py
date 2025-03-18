import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from promptcap import PromptCap
from collections import Counter
from wordcloud import WordCloud

# Define diverse prompts
PROMPTS = [
    "How many cells are in this image?",
    "Describe the density of the cells in this image.",
    "What is the dominant shape of the cells?",
    "Are the cells evenly distributed or clustered?",
    "Do the cells appear healthy or show abnormalities?",
    "Are there any artifacts or debris in the image?",
    "What factors might make segmentation difficult in this image?",
]

def load_images(folder_path):
    """Load image file paths from the given directory."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

def generate_captions(images, prompts, model):
    """Generate multiple captions per image using different prompts."""
    results = []

    for image in tqdm(images, desc="Generating captions"):
        image_captions = {"image": image, "captions": {}}

        for prompt in prompts:
            try:
                caption = model.caption(prompt, image)
                image_captions["captions"][prompt] = caption
            except Exception as e:
                image_captions["captions"][prompt] = f"Error: {str(e)}"

        results.append(image_captions)

    return results

def evaluate_captions(captions_data):
    """Evaluate captions and return scores for each prompt."""
    scores = {prompt: [] for prompt in PROMPTS}

    for entry in captions_data:
        for prompt, caption in entry["captions"].items():
            if "Error" not in caption:
                scores[prompt].append(len(caption.split()))  # Simple metric: caption length

    avg_scores = {prompt: sum(values) / len(values) if values else 0 for prompt, values in scores.items()}
    return avg_scores

def visualize_results(avg_scores):
    """Generate bar chart and word cloud from best captions."""
    # Sort prompts by score
    sorted_prompts = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    # Bar Chart Visualization
    plt.figure(figsize=(10, 5))
    plt.barh([p[0] for p in sorted_prompts], [p[1] for p in sorted_prompts], color="skyblue")
    plt.xlabel("Average Caption Length (as a quality proxy)")
    plt.ylabel("Prompt")
    plt.title("Prompt Performance Evaluation")
    plt.gca().invert_yaxis()  # Invert to show best prompts at top
    plt.show()

    # Word Cloud Visualization
    all_text = " ".join([entry["captions"].get(sorted_prompts[0][0], "") for entry in captions_data])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most Frequent Words in Best Prompt Captions")
    plt.show()

def save_results(data, output_file="captions_results.json"):
    """Save the generated captions to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def main():
    images_folder = "../../data/preprocessing_outputs/transformed_images_labels/images"
    output_file = "../../data/preprocessing_outputs/captions_results.json"

    images = load_images(images_folder)
    if not images:
        print("No images found. Exiting.")
        return

    model = PromptCap("tifa-benchmark/promptcap-coco-vqa")
    if torch.cuda.is_available():
        model.cuda()

    captions_data = generate_captions(images, PROMPTS, model)
    save_results(captions_data, output_file)

    avg_scores = evaluate_captions(captions_data)
    visualize_results(avg_scores)

    print(f"Captions saved to {output_file}. Best prompt: {max(avg_scores, key=avg_scores.get)}")

if __name__ == "__main__":
    main()
