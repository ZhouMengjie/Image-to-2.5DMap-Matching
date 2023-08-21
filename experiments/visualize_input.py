import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.getcwd())
import yaml
import pandas as pd
import numpy as np
import torch
import open3d as o3d
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from experiments.visualizer import visualize_pcd  
from openstreetmap import cropping
from data import augmentation_simple
import torchvision.transforms as transforms
import random

if __name__ == "__main__":
     # load pcd of the whole map
    data_path = 'datasets'
    area = 'unionsquare5kU'
    queries = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None).values

    # deploy augmentation on local pcd
    # idx = 20
    idx = random.randint(0, 5000)
    panoid = queries[idx][0]
    center_x = queries[idx][1]
    center_y = queries[idx][2]
    heading = queries[idx][3] # in degree
    city = queries[idx][4]

    # show panoid and tile idx
    global_queries= (pd.read_csv(os.path.join(data_path, 'csv', (area + '_set.csv')), sep=',', header=None)).values
    global_idx = global_queries[idx][1]
    print(panoid)
    print(str(global_idx).zfill(5))

    # load point cloud
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    # load semantic id 
    data = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = data.values
    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)
    # original 228*228 area
    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]
    feats = semantic_ids[vertex_indices]
    colors = []
    for i in range(len(feats)):
        class_id = feats[i][0]
        colors.append(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(colors)
    # rotation + crop
    center = [center_x, 0, center_y]
    result = {'heading': heading, 'center':center}
    result['cloud'] = torch.tensor(coords, dtype=torch.float)
    result['cloud_ft'] = torch.tensor(colors, dtype=torch.float)
    t = [augmentation_simple.RandomRotation(max_theta=0, axis=np.asarray([0,1,0])),
        augmentation_simple.RandomCenterCrop(radius=76, rnd=0)]
    transform = transforms.Compose(t)
    result = transform(result)
    new_coords = result['cloud'].numpy()
    new_colors = result['cloud_ft'].numpy()
    visualize_pcd(new_coords, new_colors, 'npy')

