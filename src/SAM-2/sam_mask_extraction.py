import time
import matplotlib.pyplot as plt
import os
import torch
import cv2
import supervision as sv
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import math

#!pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
#!pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision==0.23.0

#EXTRACTION DES MASQUES

# Définition du chemin d'accès
HOME = "src/unlabel_data_test"
print("HOME:", HOME)
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"


# Timer 1 : Chargement du modèle
t0 = time.perf_counter()
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
t1 = time.perf_counter()
print("Temps de chargement du modèle : {:.2f} s".format(t1 - t0))


def load_image(image_name):
    # Chargement de l'image
    image_path = os.path.join("data/Training/Training-labeled/images/", image_name)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb

def generate_masks(image_rgb, mask_generator):
    # Timer 2 : Génération des masques par SAM
    t2 = time.perf_counter()
    sam_result = mask_generator.generate(image_rgb)
    print("Clés du premier masque :", sam_result[0].keys())
    t3 = time.perf_counter()
    print("Temps de génération des masques : {:.2f} s".format(t3 - t2))
    return sam_result

def filter_masks_by_color(image_bgr, sam_result, hue_range=(130, 160), min_masks=2):
    """Filtre les masques basé sur la teinte moyenne des cellules avec K-means"""
    # Calcul des caractéristiques de couleur pour chaque masque
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    features = []
    
    for mask_data in sam_result:
        mask = mask_data['segmentation']
        # Calcul de la teinte moyenne dans la zone du masque
        mean_hue = np.mean(hsv[mask, 0])
        features.append([mean_hue])
    
    # Clustering K-means avec 2 groupes
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    
    # Identification du cluster violet
    cluster_hues = [kmeans.cluster_centers_[i][0] for i in range(2)]
    violet_cluster = np.argmin([abs(h - np.mean(hue_range)) for h in cluster_hues])
    
    # Sélection des masques du cluster violet
    filtered = [mask_data for i, mask_data in enumerate(sam_result) if kmeans.labels_[i] == violet_cluster]
    
    # Garantie d'au moins min_masks masques
    if len(filtered) < min_masks:
        print(f"Alerte : seulement {len(filtered)} masques violets, conservation des {min_masks} plus proches")
        hue_distances = [abs(f[0] - np.mean(hue_range)) for f in features]
        filtered_indices = np.argsort(hue_distances)[:min_masks]
        filtered = [sam_result[i] for i in filtered_indices]
    
    return filtered



def annotate_masks(image_bgr, sam_result):
    # Timer 3 : Annotation des masques
    t4 = time.perf_counter()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    t5 = time.perf_counter()
    print("Temps d'annotation des masques : {:.2f} s".format(t5 - t4))
    return annotated_image, detections

def create_grayscale_mask(image_shape, sam_masks):
    grayscale_image = np.zeros(image_shape, dtype=np.uint8)
    gray_values = [50, 150, 250]
    
    for idx, mask_data in enumerate(sam_masks):
        mask = mask_data['segmentation']
        gray_value = gray_values[idx % len(gray_values)]
        grayscale_image[mask] = gray_value
    
    return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

# Timer 4 : Création de l'image des masques en nuances de gris
def create_grayscale_mask_image(image_bgr, detections):
    t6 = time.perf_counter()
    height, width = image_bgr.shape[:2]
    grayscale_mask_image = np.zeros((height, width, 3), dtype=np.uint8)
    gray_values = [50, 150, 250]  # Valeurs de gris possibles

    for idx, mask in enumerate(detections.mask):
        gray_value = gray_values[idx % len(gray_values)]
        grayscale_mask_image[mask] = (gray_value, gray_value, gray_value)  # BGR

    # Conversion en RGB pour l'affichage (si nécessaire)
    grayscale_mask_image_rgb = cv2.cvtColor(grayscale_mask_image, cv2.COLOR_BGR2RGB)
    t7 = time.perf_counter()
    print("Temps de création du masque en nuances de gris : {:.2f} s".format(t7 - t6))
    return grayscale_mask_image_rgb

# Exemple de chemin pour l'image solution (à adapter selon votre structure)
number = "00011"
img = "cell_"+ number
image_soluce_path = os.path.join("data", "Training", "Training-labeled", "labels", f"{img}_label.png")

# Chargement de l'image source (image_bgr déjà chargée via load_image)
image_bgr, image_rgb = load_image(img + ".png")

# Traitement pour obtenir les masques, etc.
sam_result = generate_masks(image_rgb, mask_generator)
sam_result_filtered = filter_masks_by_color(image_bgr, sam_result)
annotated_all_masks, _ = annotate_masks(image_bgr, sam_result)
annotated_image, detections = annotate_masks(image_bgr, sam_result_filtered)
grayscale_mask_image = create_grayscale_mask_image(image_bgr, detections)

all_masks_grayscale = create_grayscale_mask(image_bgr.shape[:2], sam_result)
filtered_masks_grayscale = create_grayscale_mask(image_bgr.shape[:2], sam_result_filtered)

# Création de l'image de debug des teintes (hue)
hsv_display = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
hue_channel = cv2.merge([hsv_display[:,:,0],
                         np.uint8(255 * np.ones_like(hsv_display[:,:,0])),
                         np.uint8(255 * np.ones_like(hsv_display[:,:,0]))])
hue_display = cv2.cvtColor(hue_channel, cv2.COLOR_HSV2RGB)

# Charger l'image solution depuis le fichier
image_soluce = cv2.imread(image_soluce_path)

# Modifier la liste d'images
images_list = [
    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),    # Image originale
    image_soluce,                                  # Vérité terrain
    all_masks_grayscale,                           # Tous masques (gris/fond noir)
    hue_display,                                   # Carte HSV
    filtered_masks_grayscale                       # Masques filtrés (gris/fond noir)
]

titles = [
    f'Source Image ({number})',
    'Expert Annotation',
    f'Complete SAM Masks ({len(sam_result)})',
    'Spectral Analysis (Hue)',
    f'Filtered Masks (SAM + HSV) ({len(sam_result_filtered)})'
]

# Ajuster l'affichage
sv.plot_images_grid(
    images=images_list,
    grid_size=(2, 3),
    titles=titles,
    size=(18, 12)
)
plt.show()

print("end")

# image ou il serait utilile de faire SAM-2 + HSV
violet_color_image = ["1-124","277-644"]