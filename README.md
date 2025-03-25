# PM2.5 Retrieval with Sentinel-5P Data using Deep Learning

This repository contains the implementation of deep learning models for the retrieval of **PM2.5 concentrations from Sentinel-5P data over Europe**. The project implements the PMUNet, PMRes, and PMSlim architectures, with pretrained weights and the associated dataset available for download.

## Overview

The goal of this project is to estimate PM2.5 concentrations using satellite-based radiance data from Sentinel-5P. The models were trained to predict PM2.5 concentrations at ground level, leveraging state-of-the-art deep learning techniques. The project focuses on providing accurate, real-time monitoring of air quality across Europe, with potential applications for global PM2.5 retrieval.

## Models
- **PMUNet**: A convolutional neural network based on the U-Net architecture, adapted here for regression to predict PM2.5 concentrations.
- **PMRes**: A deep residual network used to improve model performance by preventing vanishing gradients.
- **PMSlim**: A lightweight model designed to offer a balance between performance and computational efficiency.

The project uses the following evaluation metrics to assess the performance of the models:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **PSNR** (Peak Signal-to-Noise Ratio)

## Pretrained Models
Pretrained weights for each model are provided, allowing for easy fine-tuning or direct application to similar datasets.

## Dataset
The dataset used for training and validation is based on Sentinel-5P data for the European region. The dataset includes cloud-free images with ground truth PM2.5 concentrations provided by CAMS[1]. The processed dataset used for the training is available for public use.

## Installation

### Requirements
- Python 3.x
- pytorch==2.1.1
- pytorch-cuda=11.8
- torchaudio==2.1.1
- torchvision==0.16.1
- tqdm
- imageio
- numpy==1.26.3

Install the necessary libraries by running:
```
conda env create -f D:\AntonioMazza\PMUnet\code\environment.yml
```
Ensure you have the necessary datasets. The dataset should be organized as follows:
```
<dataset_path>/
    Train.npy
    Val.npy
    Test.npy
    Gen.npy
```
where in these files are listed the paths of the patches you want to use for the training, validation, test and generalization sets.

## Usage

The source files are stored in the src folder. 

### Training

For the training you can use the code train.py. You can use an editor to run the code, or from command line:
```
python train.py --dataset_path <path_to_dataset> --out_folder <output_folder> --model <model_name> --exp_name <experiment_name> --gpu_number <gpu_number_if_available>
```
After the training the model will be tested on the test and generalization sets.

#### Arguments
```
--dataset_path: The path to the dataset folder containing Train.npy, Val.npy, Test.npy, and Gen.npy.

--out_folder: The folder where the experiment outputs will be stored.

--model: The model to use for training or testing. Options: PMUnet, PMRes, PMSlim.

--exp_name: The name of the experiment. If not provided, it defaults to the chosen model name.

--gpu_number: The GPU ID to use, if available.
```
### Testing

Similarly to the training code, the train.py can be run from an editor or from the command line:
```
python test.py --dataset_path <path_to_dataset> --out_folder <output_folder> --model <model_name> --exp_name <experiment_name> --gpu_number <gpu_number_if_available>
```
In the output folder the results are provided both numerically in a json file and by providing the reference (prefix: 'orig') and the concentration provided by the chosen model (prefix: 'pred').

# License 

Copyright (c) 2025 [National Biodiversity Future Center (NBFC)](https://www.nbfc.it/en), [National Research Council- Institute of Methodologies for Environmental Analysis (IMAA), Contrada Loya, Tito Scalo, 85050 Italy](https://www.cnr.it/en/institute/055/institute-of-methodologies-for-environmental-analysis-imaa).

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document `LICENSE.txt` (included in this directory).

# Citing

If you use this CNN-based approach in your research or wish to refer to the baseline results, please use the following __BibTeX__ entry.
```
@article{mazza2025,
  title={PM2.5  Retrieval with Sentinel-5P Data over Europe Exploiting Deep Learning},
  author={Mazza, Antonio and Guarino, Giuseppe and Scarpa, Giuseppe and Qiangqiang, Yuan and Gaetano, Raffaele},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={},
  number={},
  pages={},
  year={2025},
  publisher={IEEE}
}
```
