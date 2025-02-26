
# Segmentation Faiblement Supervisée des Cellules dans des Images de Microscopie Haute Résolution et Multi-modalité

## Description

Cette tâche provient d’un défi organisé lors de la conférence NeurIPS 2022. Depuis le [site du défi](https://neurips22-cellseg.grand-challenge.org/) :

La segmentation des cellules est souvent la première étape des analyses monocellulaires en biologie et recherche biomédicale basées sur les images de microscopie. L’apprentissage profond est largement utilisé pour la segmentation d’images, mais il est difficile de collecter un grand nombre d’images de cellules annotées pour entraîner les modèles, car l’annotation manuelle des cellules est extrêmement chronophage et coûteuse. De plus, les ensembles de données utilisés sont souvent limités à une seule modalité et manquent de diversité, ce qui entraîne une faible généralisation des modèles entraînés. Cette compétition vise à évaluer les méthodes de segmentation cellulaire qui pourraient être appliquées à diverses images de microscopie à travers plusieurs plateformes d’imagerie et types de tissus. Nous formulons le problème de segmentation cellulaire comme une tâche d’apprentissage faiblement supervisée pour encourager les modèles à utiliser des images étiquetées limitées et de nombreuses images non étiquetées, car les images non étiquetées sont relativement faciles à obtenir en pratique.

Cette compétition possède quatre caractéristiques principales :

* Cadre de tâche faiblement supervisée : patchs limités étiquetés + nombreuses images non étiquetées ;

* Vise à évaluer des algorithmes de segmentation cellulaire polyvalents ;

* Les images de test incluent des images de lame entière (~10,000x10,000) ;

* Métriques d’évaluation : nous nous concentrons à la fois sur la précision et l’efficacité de la segmentation.

## Données

La segmentation d’image est une tâche de classification où la classification s’effectue au niveau du pixel. Dans ce cas, le modèle doit apprendre à classer les pixels dans des images de microscopie de manière à segmenter l’image en cellules individuelles. Les images de microscopie peuvent avoir une distribution très complexe, car la structure des images peut varier considérablement selon l’échelle de zoom, le type de cellules observées ou le type de coloration appliqué au milieu. Cet ensemble de données combine des images provenant de diverses plateformes d’imagerie et types de tissus, dans une tentative d’être un échantillon représentatif des types d’imagerie microscopique observés dans le domaine de la biologie.

Cette tâche reflète l’état de nombreux problèmes actuels d’apprentissage automatique, dans la mesure où elle repose sur un mélange de données étiquetées et non étiquetées. L’étiquetage est souvent coûteux, d’où l’intérêt de la recherche pour exploiter les données non étiquetées afin de mieux apprendre la structure de la distribution des données sans les coûts supplémentaires de données étiquetées.

Un autre aspect unique de cet ensemble de données est que certaines images peuvent être extrêmement grandes, allant jusqu’à ~10,000x10,000. Cela constituera un défi computationnel, et vous devrez être créatif dans la conception d’un pipeline qui peut être efficace à ces grandes tailles.

## Travaux connexes

Minaee et al. offrent une large revue de nombreuses techniques modernes de segmentation d’images avec une comparaison de performances sur plusieurs ensembles de données [2]. Schmarje et al. passent en revue les méthodes d’apprentissage semi-, auto-, et non supervisées pour la classification d’images [3]. Il sera important de connaître certaines des architectures de vision de base telles que les réseaux convolutifs [4], U-Net [5], et Vision Transformer [6]. Concernant l’imagerie biologique, les méthodes importantes incluent [Cellpose](https://github.com/MouseLand/cellpose), [Mesmer](https://github.com/vanvalenlab/deepcell-tf), [Stardist](https://github.com/stardist/stardist), et [Omnipose](https://github.com/MouseLand/cellpose), toutes ayant des implémentations open source.

Les organisateurs de la compétition ont fourni un [répertoire GitHub](https://github.com/JunMa11/NeurIPS-CellSeg) qui démontre l’entraînement de plusieurs architectures différentes (U-Net, ViT+U-Net, et Swin Transformer+U-Net [7]) et évalue le modèle sur des données de validation. [À FAIRE : d’ici à ce que cela soit prêt pour les étudiants, les méthodes et résultats des gagnants peuvent avoir été publiés et devraient être ajoutés ici.]

## Attentes

Puisque les organisateurs de la compétition ont implémenté des bases raisonnables, il est de votre responsabilité de partir de là ! Votre objectif sera d’implémenter au moins deux nouvelles méthodes qui n’ont pas encore été appliquées à cette tâche d’imagerie microscopique. En réalité, vous devrez essayer de nombreuses combinaisons d’architectures et d’approches pour le problème de taille des images afin de déterminer ce qui fonctionne le mieux. Dans votre rapport final, nous attendons une description des méthodes essayées, avec une comparaison des avantages et inconvénients de chaque méthode tentée.

## Références

1. Site du défi : https://neurips22-cellseg.grand-challenge.org/
2. Minaee et al. (2020) Image segmentation using deep learning: a survey. [arXiv:2001.05566](https://arxiv.org/abs/2001.05566)
3. Schmarje et al. (2021) A survey on semi-, self- and unsupervised learning for image classification. *IEEE Access*. 9: 82146-82168. [doi:10.1109/ACCESS.2021.3084358](https://doi.org/10.1109/ACCESS.2021.3084358)
4. LeCun et al. (1989) Backpropagation applied to handwritten zip code recognition. *Neural Comput*. 1(4): 541-551. [doi:10.1162/neco.1989.1.4.541](https://doi.org/10.1162/neco.1989.1.4.541)
5. Ronneberger et al. (2015) U-Net: convolutional networks for biomedical image segmentation. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
6. Dosovitskiy et al. (2020) An image is worth 16x16 words: transformers for image recognition at scale. [arXiv:2010.11929v2](https://arxiv.org/abs/2010.11929v2)
7. Liu et al. (2021) Swin transformer: hierarchical vision transformer using shifted windows. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
8. [Cellpose](https://github.com/MouseLand/cellpose)
9. [Mesmer](https://github.com/vanvalenlab/deepcell-tf)
10. [Stardist](https://github.com/stardist/stardist)
11. [Omnipose](https://github.com/MouseLand/cellpose)
