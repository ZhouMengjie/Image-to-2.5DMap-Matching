# Image-based Geolocalization by Ground-to-2.5D Map Matching
This repository is an official implementation of our latest work: 
- [Image-based Geolocalization by Ground-to-2.5D Map Matching](https://arxiv.org/abs/2308.05993 "Image-based Geolocalization by Ground-to-2.5D Map Matching")

The main task of our work involves querying a ground-view image in relation to a large-scale and highly detailed georeferenced map, which consists of 2.5D structural map models and 2D aerial-view map images. An illustration of the work is shown below.

## Data Preperation
We provide a [large-scale ground-to-2.5D map geolocalization dataset](https://github.com/ZhouMengjie/2-5DMap-Dataset):
- Panoramic images and map tiles for training and testing can be directly obtained via the provided link.
- 2.5D maps for testing can also be directly downloaded with the provided link.
- 2.5D maps for training have to be processed parallel using the provided code and metadata.
- Prepared datasets should be stored in ./datasets directory.

## Codes
We offer a Python implementation to learn location embeddings of panoramic images and multi-modal maps.

### Prerequisite
Here are the commands to configure your own environment:
```
conda env create -f environment.yml
```

### Training
```
sh trains.sh
```
- In this file, you can set the type of task (single-modal or multi-modal), model, optimizer, loss, fusion strategy and training hyperparameters.
- Refer to ./config and ./training/train.py for more details about the configuration.

### Evaluation
```
sh evals.sh
```
- In this file, you can evaluate the performance of single-image based localization for different testing areas. 
- To achieve [route-based localization](https://github.com/ZhouMengjie/you-are-here), you should output the location embeddings of query images and reference maps.
- Please comment out the corresponding codes in ./eval/evaluate_pickle.py to generate the required files.

## Weights

## Results

## Disclaimer
We make no claims about the stability or usability of the code provided in this repository.
We provide no warranty of any kind, and accept no liability for damages of any kind that result from the use of this code.


## Citation
If you have any questions, please feel free to leave a message or contact me via "mengjie.zhou@bristol.ac.uk". If you use this code, please cite:
```latex
@article{zhou2023image,
  title={Image-based Geolocalization by Ground-to-2.5D Map Matching},
  author={M. Zhou, L. Liu, Y. Zhong, A. Calway},
  journal={arXiv preprint arXiv:2308.05993},
  year={2023}
}
```






