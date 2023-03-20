# Pdf-Enhancer
Project on enhancing the pdf by super resolution and image stitching

## Update:
This pipeline of super resolution and image stitching can also be used in making high quality panorama images and also for satellite and drone images.

## Our Approach
 * The pipeline involves the application of super resolution of the uploaded pdf images
 * The enhanced images are then stiched into one image as pdf
 
## Problem Statement
Conversion of low quality pdf images into high quality single image pdf. It has a high usability in various purposes.

## Unsupervised Deep Image Stitching
Traditional feature-based image stitching technologies rely heavily on feature detection quality, often failing to stitch images with few features or low resolution. The learning-based image stitching solutions are rarely studied due to the lack of labeled data, making the supervised methods unreliable. To address the above limitations, we propose an unsupervised deep image stitching framework consisting of two stages: unsupervised coarse image alignment and unsupervised image reconstruction. In the first stage, we design an ablation-based loss to constrain an unsupervised homography network, which is more suitable for large-baseline scenes. Moreover, a transformer layer is introduced to warp the input images in the stitching-domain space. In the second stage, motivated by the insight that the misalignments in pixel-level can be eliminated to a certain extent in feature-level, we design an unsupervised image reconstruction network to eliminate the artifacts from features to pixels. Specifically, the reconstruction network can be implemented by a low-resolution deformation branch and a high-resolution refined branch, learning the deformation rules of image stitching and enhancing the resolution simultaneously. To establish an evaluation benchmark and train the learning framework, a comprehensive real-world image dataset for unsupervised deep image stitching is presented and released. Extensive experiments well demonstrate the superiority of our method over other state-of-the-art solutions. Even compared with the supervised solutions, our image stitching quality is still preferred by users.

## Papers reviewed
* [Unsupervised Deep Image Stitching](https://arxiv.org/pdf/2106.12859)
* [Pixel-wise Deep Image Stitching](https://arxiv.org/pdf/2112.06171)
* [Deep Rectangling for Image Stitching](https://arxiv.org/pdf/2203.03831)
