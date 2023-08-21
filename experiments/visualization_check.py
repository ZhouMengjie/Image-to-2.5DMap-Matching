import os
import sys
import torch
sys.path.append(os.getcwd())
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import yaml
import random
import torchvision.transforms as transforms
import numpy as np
from experiments.visualizer import visualize_pcd  

def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)


def projection(center, pc, image_h, image_w, sensor_h, sensor_w, npoints, batch_size):
        center_x = center[:,0]
        center_y = center[:,2]
        cx = 0.5*sensor_w - center_x
        cy = 0.5*sensor_h - center_y
        cx = cx.view(batch_size, 1)
        cy = cy.view(batch_size, 1)
        image_x = (pc[:,0,:] + cx)*(image_w - 1) / (sensor_w - 1)
        image_y = (pc[:,2,:] + cy)*(image_h - 1) / (sensor_h - 1)
        return torch.cat([
            image_x[:,None,:],
            image_y[:,None,:],
        ], dim=1)

if __name__ == "__main__":
    data_folder = 'train_batch'
    seed = 0
    tile_embedding = np.load(os.path.join(data_folder, 'tile_embedding.npy'))
    center = np.load(os.path.join(data_folder, 'center.npy'))
    xyz = np.load(os.path.join(data_folder, 'xyz.npy'))
    visualize_pcd(xyz[seed].T, None, 'npy')
    tile_embedding = torch.tensor(tile_embedding)
    center = torch.tensor(center)
    xyz = torch.tensor(xyz)
    batch_size, _, image_h, image_w = tile_embedding.shape
    image = projection(center,xyz,image_h,image_w,156,156,1024,batch_size) 

    # visualize coordinates and maps
    coords = np.load(os.path.join(data_folder, 'coords.npy'))
    labels = np.load(os.path.join(data_folder, 'labels.npy'))
    visualize_pcd(coords[seed].T, None, 'npy')

    area = 'hudsonriver5kU'
    data = pd.read_csv(os.path.join('datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info = data.values
    idx = labels[seed]
    global_idx = info[idx][1] # for train only
    city = info[idx][0]
    tile_pathname = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_pathname)
    tile.show()

    # visualiz feature map
    image = image.type(torch.int)
    image_map = image[seed].numpy()
    feature_map = tile_embedding[seed].numpy()
    feature_map_combination = []
    plt.figure()
 
    for i in range(0, feature_map.shape[0]):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)
    feature_map_sum = sum(ele for ele in feature_map_combination)

    plt.imshow(feature_map_sum)
    point_list = image_map
    for i in range(1024):
        plt.scatter(point_list[0][i],point_list[1][i], s=5, c='r')
    plt.show()

    # visualize original tile
    plt.figure()
    tiles = np.load(os.path.join(data_folder, 'tiles.npy'))
    batch_size, _, image_h, image_w = tiles.shape
    tile_tensor = torch.tensor(tiles[seed])
    tile = tensor2img(tile_tensor)
    image = projection(center,xyz,image_h,image_w,156,156,1024,batch_size) 
    image = image.type(torch.int)
    image_map = image[seed].numpy()

    plt.imshow(np.array(tile))
    point_list = image_map
    for i in range(1024):
        plt.scatter(point_list[0][i],point_list[1][i], s=5, c='r')
    plt.show()



   
    





