# Weakly Supervised Cell Segmentation in Multi-modality High-resolution Microscopy Images

## Description

This task comes from a challenge that ran during the 2022 NeurIPS conference. From the [challenge website](https://neurips22-cellseg.grand-challenge.org/):

Cell segmentation is usually the first step for downstream single-cell analysis in microscopy image-based biology and biomedical research. Deep learning has been widely used for image segmentation, but it is hard to collect a large number of labeled cell images to train models because manually annotating cells is extremely time-consuming and costly. Furthermore, datasets used are often limited to one modality and lacking in diversity, leading to poor generalization of trained models. This competition aims to benchmark cell segmentation methods that could be applied to various microscopy images across multiple imaging platforms and tissue types. We frame the cell segmentation problem as a weakly supervised learning task to encourage models that use limited labeled and many unlabeled images for cell segmentation as unlabeled images are relatively easy to obtain in practice.

This competition has four main features:

* Weakly supervised task setting: limited labeled patches + many unlabeled images

* Aims to benchmark vers atile cell segmentation algorithms;

* Testing images include whole-slide image (~10,000x10,000);

* Evaluation metrics: we focus on both segmentation accuracy and efficiency.

## Data

Image segmentation is a classification task where the classification happens at the pixel level. In this case, the model needs to learn to classify the pixels in microscopy images so that the classifications segment the image into individual cells. Microscopy images can have a very complex distribution, because the image structure may differ drastically depending on the scale of zoom, the kinds of cells being observed, or the kind of coloring applied to the medium. This dataset combines imaging from a variety of imaging platforms and tissue types, in an attempt to be a representative sample of the kinds of microscopy seen across the field of biology.

This task reflects the state of many current machine learning problems, in that it relies on a mix of unlabeled and labeled data. Labeling is often expensive, hence the research interest in leveraging unlabeled data to better learn the structure of the data distribution without the expense of more labeled data.

Another unique aspect of this dataset is that some images can be extremely large, up to ~10,000x10,000. This will prove a computational challenge, and you will need to be creative in designing a pipeline that can be effective at these large sizes.

## Related work

Minaee et al. provide a broad review of many modern image segmentation techniques with a comprehensive performance comparison over several datasets [2]. Schmarje et al. survey semi-, self-, and unsupervised learning methods for image classification [3]. It will be important to know about some of the basic vision architectures like convolutional networks [4], U-Net [5], and Vision Transformer [6]. Relevant to biological imaging, important methods include [Cellpose](https://github.com/MouseLand/cellpose), [Mesmer](https://github.com/vanvalenlab/deepcell-tf), [Stardist](https://github.com/stardist/stardist), and [Omnipose](https://github.com/MouseLand/cellpose), all of which have open source implementations.

The competition hosts provided a [GitHub repository](https://github.com/JunMa11/NeurIPS-CellSeg) that demonstrates training several different architectures (U-Net, ViT+U-Net, and Swin Transformer+U-Net [7]) and evaluating the model on validation data. [TODO: by the time this is ready for students, the methods and results of the winners may have been released and should be added here.]

## Expectations

Since the competition hosts have implemented reasonable baselines, it is your job to build from there! Your goal will be to implement at least two new methods that have not yet been applied to this microscopic imaging task. In reality, you will need to try many combinations of architectures and approaches to the image size problem to see what works the best. In your final report we expect a description of the methods you tried, with a comparison of the advantages and shortcomings of each attempted method.

## References

1. Challenge website: https://neurips22-cellseg.grand-challenge.org/
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