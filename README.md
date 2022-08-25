# GANomaly in Pytorch

This is an unofficial repository of [GANomaly Paper Review: Semi-Supervised Anomaly Detection via Adversarial Training](https://arxiv.org/abs/1805.06725). To have a fast overview, there is also a review I wrote about the paper [here](https://medium.com/p/a6f7a64a265f).

## Pipeline 

![image](https://miro.medium.com/max/1400/1*kpZKFb8l-TIRC9SVB2ET_w.png)

 There are two encoders, a decoder and a discriminative model included in this approach.

* **Autoencoder**, which is also the Generator of the model, learns to reconstruct the original input by using the encoder and decoder networks.
* **Discriminator** is trained to distinguish the inputs (true samples) from the reconstructions (false samples).
* The second **Encoder** comprises the reconstruction into a latent code $\hat{z}$.

## MVTec dataset

* The MVTec Anomaly Detection dataset contains 15 categories with 3629 high-resolution images for training and 1725 images for testing. 
* The training set contains only images without defects 
* The test set contains both normal images and images containing different types of defects
* All image resolutions are in the range between 700 x 700 and 1024 x 1024 pixels.

![image](https://user-images.githubusercontent.com/61031596/175270521-a0829113-fa8b-493f-b28e-b8c0bf129d3b.png)

For other informations read this [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)

## Installation

1. Clone repository

```
git clone https://github.com/eugeniaring/GANomaly-MVTec-AD.git
````

2. Download dataset

3. Create a virtual environment in Python

4. Install the requirements

```
pip install -r requirements.txt
````
5. Train and evaluate GANomaly on MVTec AD

```
python train.py
````

## Results 

Below there are the image-level ROC-AUC scores:

|  | Paper    | Implementation    |
| :---:   | :---: | :---: |
| bottle | -   | 0.813   |
| cable | -   |    |
| capsule | -   |    |
| carpet | -   |    |
| grid | -   |    |
| hazelnut | -   |    |
| leather | -   |    |
| metal_nut | -   |    |
| pill | -   |    |
| screw | -   |    |
| tile | -   |    |
| toothbrush | -   |    |
| transistor | -   |    |
| wood | -   |    |
| zipper | -   |    |
| Average | -   |    |



