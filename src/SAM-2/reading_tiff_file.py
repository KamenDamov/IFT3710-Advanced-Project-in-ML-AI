import time
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import torch
import cv2
import supervision as sv
import numpy as np
from PIL import Image


def convert_images_to_png(folder_path, delete_original=True):
    """
    Convertit tous les fichiers .tiff, .tif et .bmp d'un dossier en PNG.
    Si delete_original est True, les fichiers d'origine sont supprimés après conversion.
    """
    # Parcours de tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # On ne traite que les fichiers
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            new_filename = os.path.splitext(filename)[0] + ".png"
            new_file_path = os.path.join(folder_path, new_filename)
            try:
                if ext in ['.tiff', '.tif']:
                    # Lecture du fichier TIFF
                    img = tiff.imread(file_path)
                    # Si l'image contient plusieurs pages, sélectionnez la première
                    img_to_save = img[0] if img.ndim > 2 else img
                    # Sauvegarde en PNG
                    plt.imsave(new_file_path, img_to_save, cmap='gray')
                    print(f"Converti : {filename} -> {new_filename}")
                elif ext == '.bmp':
                    # Conversion des fichiers BMP en PNG avec Pillow
                    img = Image.open(file_path)
                    img.save(new_file_path, "PNG")
                    print(f"Converti : {filename} -> {new_filename}")
                else:
                    continue
                # Suppression du fichier original si demandé
                if delete_original:
                    os.remove(file_path)
                    print(f"Fichier original supprimé : {filename}")
            except Exception as e:
                print(f"Erreur lors de la conversion de {filename} : {e}")


# Exemple d'utilisation :
# folders = ["data/Training/Training-labeled/images","data/Training/Training-labeled/labels","data/Training/train-unlabeled-part1","data/Training/train-unlabeled-part2"]
# convert_images_to_png(folders[3])


