import os
import numpy as np
from skimage import io, color, measure
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label

def load_and_convert_to_mask(image_path):
    """
    Charge une image de segmentation et la convertit en masque d'instances
    """
    # Charger l'image
    img = io.imread(image_path)
    
    # Si l'image est en couleur, la convertir en niveaux de gris
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = color.rgb2gray(img)
    
    # Binariser l'image si nécessaire (seuil à ajuster selon vos images)
    binary = img > 0
    
    # Étiqueter les composantes connexes pour créer un masque d'instances
    labeled_mask, num_features = label(binary)
    
    return labeled_mask

def evaluate_f1_score(masks_true, masks_pred, threshold=0.5):
    """
    Calcule le score F1, la précision et le rappel en utilisant l'IOU entre les instances
    """
    true_objects = np.unique(masks_true)
    pred_objects = np.unique(masks_pred)
    
    # Éliminer l'étiquette 0 (fond)
    true_objects = true_objects[true_objects != 0]
    pred_objects = pred_objects[pred_objects != 0]
    
    # Si l'un des masques est vide, retourner des métriques nulles
    if len(true_objects) == 0 or len(pred_objects) == 0:
        return 0, 0, 0
    
    # Calculer l'IOU pour chaque paire d'instances
    iou_matrix = np.zeros((len(true_objects), len(pred_objects)))
    
    for i, true_obj in enumerate(true_objects):
        true_mask = masks_true == true_obj
        for j, pred_obj in enumerate(pred_objects):
            pred_mask = masks_pred == pred_obj
            
            # Intersection
            intersection = np.logical_and(true_mask, pred_mask).sum()
            # Union
            union = np.logical_or(true_mask, pred_mask).sum()
            
            # IOU
            iou = intersection / union if union > 0 else 0
            iou_matrix[i, j] = iou
    
    # Trouver les correspondances entre les instances utilisant l'algorithme hongrois
    matched_indices = []
    if iou_matrix.size > 0:
        true_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = list(zip(true_indices, pred_indices))
    
    # Compter les TP, FP, FN
    tp = 0
    for true_idx, pred_idx in matched_indices:
        if iou_matrix[true_idx, pred_idx] >= threshold:
            tp += 1
    
    fp = len(pred_objects) - tp
    fn = len(true_objects) - tp
    
    # Calculer la précision, le rappel et le F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_segmentation_models(gt_folder, baseline_folder, sam2_folder):
    """
    Évalue les performances de deux modèles de segmentation en calculant les métriques F1, précision et rappel.
    """
    # Liste des fichiers d'images dans le dossier de vérité terrain
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    
    baseline_metrics = {'precision': [], 'recall': [], 'f1_score': [], 'image_name': []}
    sam2_metrics = {'precision': [], 'recall': [], 'f1_score': [], 'image_name': []}
    
    for gt_file in gt_files:
        print(f"Traitement de l'image {gt_file}")
        
        # Charger et convertir l'image de vérité terrain en masque
        gt_path = os.path.join(gt_folder, gt_file)
        mask_true = load_and_convert_to_mask(gt_path)
        
        # Charger et évaluer l'image segmentée par le modèle baseline
        baseline_path = os.path.join(baseline_folder, gt_file)
        if os.path.exists(baseline_path):
            mask_baseline = load_and_convert_to_mask(baseline_path)
            precision, recall, f1_score = evaluate_f1_score(mask_true, mask_baseline)
            baseline_metrics['precision'].append(precision)
            baseline_metrics['recall'].append(recall)
            baseline_metrics['f1_score'].append(f1_score)
            baseline_metrics['image_name'].append(gt_file)
            print(f"  Baseline - P: {precision:.4f}, R: {recall:.4f}, F1: {f1_score:.4f}")
        
        # Charger et évaluer l'image segmentée par le modèle SAM2
        sam2_path = os.path.join(sam2_folder, gt_file)
        if os.path.exists(sam2_path):
            mask_sam2 = load_and_convert_to_mask(sam2_path)
            precision, recall, f1_score = evaluate_f1_score(mask_true, mask_sam2)
            sam2_metrics['precision'].append(precision)
            sam2_metrics['recall'].append(recall)
            sam2_metrics['f1_score'].append(f1_score)
            sam2_metrics['image_name'].append(gt_file)
            print(f"  SAM2 - P: {precision:.4f}, R: {recall:.4f}, F1: {f1_score:.4f}")
    
    # Calculer les moyennes des métriques
    results = {
        'baseline': {
            'precision': np.mean(baseline_metrics['precision']),
            'recall': np.mean(baseline_metrics['recall']),
            'f1_score': np.mean(baseline_metrics['f1_score'])
        },
        'sam2': {
            'precision': np.mean(sam2_metrics['precision']),
            'recall': np.mean(sam2_metrics['recall']),
            'f1_score': np.mean(sam2_metrics['f1_score'])
        }
    }
    
    return results, baseline_metrics, sam2_metrics

def visualize_results(results, baseline_metrics, sam2_metrics):
    """
    Visualise les résultats avec des graphiques pour faciliter la comparaison
    """
    # Graphique des moyennes
    plt.figure(figsize=(12, 6))
    
    labels = ['Précision', 'Rappel', 'F1-Score']
    baseline_means = [results['baseline']['precision'], results['baseline']['recall'], results['baseline']['f1_score']]
    sam2_means = [results['sam2']['precision'], results['sam2']['recall'], results['sam2']['f1_score']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_means, width, label='Baseline')
    plt.bar(x + width/2, sam2_means, width, label='SAM2')
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.3)
    
    plt.xlabel('Métriques')
    plt.ylabel('Score')
    plt.title('Comparaison des performances moyennes')
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison_means.png')
    
    # Graphique de comparaison F1 par image
    plt.figure(figsize=(15, 8))
    
    common_images = set(baseline_metrics['image_name']).intersection(set(sam2_metrics['image_name']))
    image_indices = range(len(common_images))
    
    baseline_f1_values = []
    sam2_f1_values = []
    common_image_names = []
    
    for img in common_images:
        baseline_idx = baseline_metrics['image_name'].index(img)
        sam2_idx = sam2_metrics['image_name'].index(img)
        
        baseline_f1_values.append(baseline_metrics['f1_score'][baseline_idx])
        sam2_f1_values.append(sam2_metrics['f1_score'][sam2_idx])
        common_image_names.append(img)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(common_image_names)), baseline_f1_values, 'o-', label='Baseline')
    plt.plot(range(len(common_image_names)), sam2_f1_values, 's-', label='SAM2')
    plt.xticks(range(len(common_image_names)), common_image_names, rotation=90)
    plt.xlabel('Images')
    plt.ylabel('F1-Score')
    plt.title('Comparaison des F1-Scores par image')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparison_by_image.png')

def print_evaluation_results(results):
    """
    Affiche les résultats d'évaluation des modèles.
    """
    print("\n=== Résultats d'évaluation des modèles ===")
    print("\nBaseline Model:")
    print(f"  Précision moyenne: {results['baseline']['precision']:.4f}")
    print(f"  Rappel moyen: {results['baseline']['recall']:.4f}")
    print(f"  F1-Score moyen: {results['baseline']['f1_score']:.4f}")
    
    print("\nSAM2 Model:")
    print(f"  Précision moyenne: {results['sam2']['precision']:.4f}")
    print(f"  Rappel moyen: {results['sam2']['recall']:.4f}")
    print(f"  F1-Score moyen: {results['sam2']['f1_score']:.4f}")
    
    # Déterminer le meilleur modèle basé sur le F1-Score
    if results['baseline']['f1_score'] > results['sam2']['f1_score']:
        diff = results['baseline']['f1_score'] - results['sam2']['f1_score']
        print(f"\n➤ Le modèle Baseline performe mieux avec un F1-Score supérieur de {diff:.4f}.")
    elif results['sam2']['f1_score'] > results['baseline']['f1_score']:
        diff = results['sam2']['f1_score'] - results['baseline']['f1_score']
        print(f"\n➤ Le modèle SAM2 performe mieux avec un F1-Score supérieur de {diff:.4f}.")
    else:
        print("\n➤ Les deux modèles ont des performances équivalentes.")

if __name__ == "__main__":
    # Définir les chemins vers les dossiers
    ground_truth_folder = "data/unified_set_rename/labels"
    baseline_folder = "data/baselineLabelsProcessed"
    sam2_folder = "data/sam2Labels" #"data/baselineLabelsProcessed"
    
    # Calculer les métriques
    results, baseline_details, sam2_details = evaluate_segmentation_models(
        ground_truth_folder, baseline_folder, sam2_folder
    )
    
    # Afficher les résultats
    print_evaluation_results(results)
    
    # Visualiser les résultats
    visualize_results(results, baseline_details, sam2_details)
    
    # Créer un DataFrame pour les métriques détaillées par image
    baseline_df = pd.DataFrame(baseline_details)
    sam2_df = pd.DataFrame(sam2_details)
    
    # Fusionner les DataFrames sur le nom de l'image
    merged_df = pd.merge(baseline_df, sam2_df, on='image_name', suffixes=('_baseline', '_sam2'))