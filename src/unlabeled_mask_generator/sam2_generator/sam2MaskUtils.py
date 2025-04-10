import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import csv


# NB : les mask générer sont en noir et blanc actuellement et non en nuance de gris
# --- Fonctions utilitaires ---

def prepare_mask(mask):
    """Convertit un masque (RGB/RGBA ou 2D) en binaire 2D."""
    if len(mask.shape) > 2:
        if mask.shape[2] == 3:  # RGB
            gray_mask = np.mean(mask, axis=2)
        elif mask.shape[2] == 4:  # RGBA
            gray_mask = np.mean(mask[:, :, :3], axis=2)
        else:
            gray_mask = mask[:, :, 0]
        return gray_mask > 0
    else:
        return mask > 0

def calculate_f1(mask1, mask2):
    """Calcule le F1 score entre deux masques binaires."""
    mask1_bin = prepare_mask(mask1)
    mask2_bin = prepare_mask(mask2)
    return f1_score(mask1_bin.flatten(), mask2_bin.flatten())

def create_grayscale_mask(anns, borders=True):
    """
    Crée un masque binaire (fond noir et zones de masque en blanc)
    à partir de tous les masques.
    """
    if len(anns) == 0:
        return None
    
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    h, w = sorted_anns[0]['segmentation'].shape
    
    # Création d'une image en niveaux de gris (ici binaire)
    mask_image = np.zeros((h, w), dtype=np.uint8)
    
    # Pour chaque masque, on met à 255 (blanc) tous les pixels couverts
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_image[m] = 255
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(mask_image, contours, -1, 255, thickness=1)
    
    return mask_image

def filter_masks_by_color_sam2(image_rgb, sam_result, hue_range=(130, 160), min_masks=2, sat_threshold=40):
    """
    Version améliorée pour SAM2 avec :
    - Gestion native de l'espace colorimétrique RGB
    - Filtrage supplémentaire par saturation
    - Regroupement intelligent des teintes circulaires
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    features = []
    valid_masks = []
    
    # Pré-filtrage par saturation
    for mask_data in sam_result:
        mask = mask_data['segmentation']
        mean_sat = np.mean(hsv[mask, 1])
        if mean_sat > sat_threshold:
            valid_masks.append(mask_data)
            mean_hue = np.mean(hsv[mask, 0])
            features.append([mean_hue])
    
    if not valid_masks:
        return []
    
    features = np.array(features)
    features_rad = np.deg2rad(features * 2)  # Conversion en radians [0-360°]
    features_circ = np.column_stack([np.sin(features_rad), np.cos(features_rad)])
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(features_circ)
    centroids_rad = np.arctan2(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    centroids_deg = np.rad2deg(centroids_rad) % 180
    
    target_hue = np.mean(hue_range)
    hue_distances = np.abs(centroids_deg - target_hue)
    violet_cluster = np.argmin(hue_distances)
    
    filtered = [valid_masks[i] for i, label in enumerate(kmeans.labels_) if label == violet_cluster]
    
    if len(filtered) < min_masks:
        hue_scores = 1 - np.abs(features[:, 0] - target_hue) / 180
        best_indices = np.argsort(hue_scores)[-min_masks:]
        filtered = [valid_masks[i] for i in best_indices]
    
    return filtered

def create_grayscale_mask_sam2(sam_masks, image_shape, border_size=0):
    """
    Crée un masque binaire (fond noir, masque blanc) à partir des masques filtrés.
    """
    grayscale_image = np.zeros(image_shape[:2], dtype=np.uint8)
    # On ne différencie plus les masques par des nuances, on met tout en blanc.
    sorted_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
    
    for mask_data in sorted_masks:
        mask = mask_data['segmentation']
        grayscale_image[mask] = 255
        if border_size > 0:
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(grayscale_image, contours, -1, 255, border_size)
    
    return grayscale_image


""" 
Exemple : 

image_Number = "01347"
image = np.array(Image.open(f"data/unified_set_rename/images/cell_{image_Number}.png").convert("RGB"))
imageSoluce = np.array(Image.open(f"data/unified_set_rename/labels/cell_{image_Number}_label.tiff").convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

device = "cpu"  # ou "cpu" selon votre configuration
sam2_checkpoint = "src/models/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

masks = mask_generator.generate(image)
filtered_masks = filter_masks_by_color_sam2(
    image,
    masks,
    hue_range=(120, 170),  # De bleu sombre (120) à violet (170)
    sat_threshold=30,      # Seuil de saturation plus bas pour inclusions
    min_masks=3            # Garantie d'au moins 3 masques
)
# Création des masques binaires
binary_mask_all = create_grayscale_mask(masks)
binary_mask_filtered = create_grayscale_mask_sam2(filtered_masks, image.shape)

# Affichage pour vérification
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title(f'Image Originale (cell_{image_Number})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_mask_all, cmap='gray')
plt.title(f'Tous les Masques en Blanc\n(cell_{image_Number})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(binary_mask_filtered, cmap='gray')
plt.title(f'Masques Filtrés en Blanc\n(cell_{image_Number})')
plt.axis('off')

plt.tight_layout()
plt.show()
"""