# Image-based Geolocalization by Ground-to-2.5D Map Matching
This repository is an official implementation of our latest work: (to be published, arxiv version)
- [Image-based Geolocalization by Ground-to-2.5D Map Matching](https://arxiv.org/abs/2308.05993 "Image-based Geolocalization by Ground-to-2.5D Map Matching")

The main task of our work involves querying a ground-view image in relation to a large-scale and highly detailed georeferenced map, which consists of 2D map tiles and 2.5D map models. An illustration of the task is shown below. Besides, an animation can be viewed in video.mp4.

![introduction.pdf](https://github.com/ZhouMengjie/Image-to-2.5DMap-Matching/files/14613547/introduction.pdf)

# Data Preperation
We provide a [large-scale ground-to-2.5D map geolocalization dataset](https://github.com/ZhouMengjie/2-5DMap-Dataset):
- Panoramic images and map tiles for training and testing can be directly obtained via the provided link.
- 2.5D maps for testing can also be directly downloaded via the provided link.
- 2.5D maps for training have to be processed parallel using the provided metadata and code.
- Prepared datasets will be stored in the following directory structure:
```
|–– datasets
|   |––csv
|   |––manhattan
|   |––pittsburgh
|   |––jpegs_manhattan_2019
|   |––jpegs_pittsburgh_2019
|   |––tiles_manhattan_2019
|   |––tiles_pittsburgh_2019
|   |––trainstreetlearnU_cmu5kU_idx
|   |––hudsonriver5kU_idx
|   |––wallstreet5kU_idx
|   |––unionsquare5kU_idx
```

## Codes
We offer a Python implementation to learn location embeddings of panoramic images and multi-modal maps. An illustration of the network architecture is shown below. 

![model.pdf](https://github.com/ZhouMengjie/Image-to-2.5DMap-Matching/files/14613554/model.pdf)

### Prerequisite
Here are the commands to configure your own environment:
```
conda env create -f environment.yml
```
You can also choose to install the necessary packages individually:
```
conda create -n env_name python=3.8
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install colour_demosaicing==0.2.1
pip3 install pytorch-metric-learning==1.3.0
pip3 install tensorflow==2.8.0
pip3 install pyyaml
pip3 install pandas==1.4.2
pip3 install opencv-python==4.5.5.64
pip3 install tqdm==4.64.0
pip3 install sklearn==0.0
pip3 install matplotlib==3.4.3
pip3 install protobuf==3.20.0
pip3 install timm==0.6.13
pip3 install open3d==0.15.1
```

### Training
```
sh trains.sh
```
- In this file, you can set the type of task (single-modal or multi-modal), model, optimizer, loss function, fusion strategy and training hyperparameters.
- Refer to ./config and ./training/train.py for more details about the configuration.

### Evaluation
```
sh evals.sh
```
- In this file, you can evaluate the performance of single-image based localization for different testing areas. 
- To achieve [route-based localization](https://github.com/ZhouMengjie/you-are-here), you should output the location embeddings of query images and reference maps.
- Please comment out the corresponding code in ./eval/evaluate_pickle.py to generate the required files.

### Weights
- Pretrained weights can be required via this [link](https://drive.google.com/drive/folders/17dWlMpof-ii6eV1KswQdSh5p3SrpIAUm?usp=sharing).
- The naming convention of each checkpoint is "model2d_model3d_optimizer_fusion.pth".

## Results
- Code to plot figures and generate videos can be found in ./experiments.
- Here, we show the performance comparison between single-modal and multi-modal methods in the single-image based localization task.
- Video showing dynamic localization procedures can be watched in supplementrary.mp4.

## Disclaimer
We make no claims about the stability or usability of the code provided in this repository.
We provide no warranty of any kind, and accept no liability for damages of any kind that result from the use of this code.


## Citation
If you have any questions, please feel free to leave a message or contact me via "mengjie.zhou@bristol.ac.uk" or "mengjiezhou2018@outlook.com". If you use this code, please cite:
```latex
@article{zhou2023image,
  title={Image-based Geolocalization by Ground-to-2.5D Map Matching},
  author={M. Zhou, L. Liu, Y. Zhong, A. Calway},
  journal={arXiv preprint arXiv:2308.05993},
  year={2023}
}
```






