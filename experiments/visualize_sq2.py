import os
import sys
sys.path.append(os.getcwd())
import cv2
import torch
import torchvision.transforms as transforms
import random
import pandas as pd
import numpy as np
from PIL import Image
# import open3d as o3d
import yaml
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from data import augmentation_simple


def farthest_point_sample(point, feature, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N) # same as shuffle
    xyz2 = np.sum(xyz ** 2, -1) 
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = -2 * np.matmul(xyz, centroid)
        dist += xyz2
        dist +=  np.sum(centroid ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pt = point[centroids.astype(np.int32)]
    ft = feature[centroids.astype(np.int32)]
    return pt, ft


if __name__ == '__main__':
    dataset_path = 'datasets'
    query_filename = 'wallstreet5kU'
    # sequences
    seq_filepath = os.path.join(dataset_path, 'csv', query_filename+'_sq.csv')
    sequences = (pd.read_csv(seq_filepath, sep=',', header=None)).values
    # pano meta
    meta_filepath = os.path.join(dataset_path, 'csv', query_filename+'_nb.csv')
    pano_meta = (pd.read_csv(meta_filepath, sep=',', header=None)).values
    # map meta
    set_filepath = os.path.join(dataset_path, 'csv', query_filename+ '_set.csv')
    map_meta = (pd.read_csv(set_filepath, sep=',', header=None)).values

    # load all points and semantic labels
    city = 'manhattan'
    points = np.load(os.path.join(dataset_path, city, city+'U.npy'))
    classes = pd.read_csv(os.path.join(dataset_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = classes.values
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    # Random choose a sequence to visualize
    # ndx = random.randint(1, len(sequences))
    ndx = 1168

    # Define the folder where to save the images
    output_folder = os.path.join('sequence_images',query_filename, str(ndx))
    os.makedirs(output_folder, exist_ok=True)

    seq_indices = sequences[ndx]
    
    for i, idx in enumerate(seq_indices):
        panoid = pano_meta[idx][0]
        yaw = float(pano_meta[idx][3])
        city = pano_meta[idx][4]
        pano_pathname = os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid+'.jpg')
        pano_image = Image.open(pano_pathname)
        # Construct the new filenames
        pano_filename = f'{ndx}_pano_{i}_{yaw}.png'
        # Save the images to the output folder with corrected names
        pano_image.save(os.path.join(output_folder, pano_filename))
        # Close the images to free up resources
        pano_image.close()

        # multi-modal map
        center_x = pano_meta[idx][1]
        center_y = pano_meta[idx][2]
        heading = pano_meta[idx][3]
        vertex_indices = np.load(os.path.join(dataset_path, (query_filename+'_idx'), (panoid + '.npy')))
        coords = points[vertex_indices][:]
        feats = semantic_ids[vertex_indices]
        colors = []
        for j in range(len(feats)):
            class_id = feats[j][0]
            colors.append(np.divide(color['color_map'][class_id], 255))
        colors = np.asarray(colors)   
        pc = torch.tensor(coords, dtype=torch.float)
        center = [center_x, 0, center_y]
        result = {'heading': heading, 'center':center}
        result['cloud'] = torch.tensor(coords, dtype=torch.float)
        result['cloud_ft'] = torch.tensor(colors, dtype=torch.float)
        t = [augmentation_simple.RandomRotation(max_theta=0, axis=np.asarray([0,1,0])),
            augmentation_simple.RandomCenterCrop(radius=76, rnd=0)]
        transform = transforms.Compose(t)    
        result = transform(result)
        coords = result['cloud'].numpy()
        feats = result['cloud_ft'].numpy()
        coords, feats = farthest_point_sample(coords, feats, 1024)
        result['cloud'] = torch.tensor(coords, dtype=torch.float)
        result['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
        new_coords = result['cloud'].numpy()
        new_colors = result['cloud_ft'].numpy()

        # 3D to 2D projection
        center_x = torch.tensor(center_x)
        center_y = torch.tensor(center_y)
        image_w, image_h = 256, 256
        sensor_w, sensor_h = 152, 152
        pc = result['cloud']   # N x C
        cx = 0.5*sensor_w - center_x + 1
        cy = 0.5*sensor_h - center_y + 1
        image_x = torch.round((pc[:,0] + cx)*(image_w - 1) / (sensor_w))
        image_y = torch.round((pc[:,2] + cy)*(image_h - 1) / (sensor_h))
        image = torch.cat([
            image_x[:,None],
            image_y[:,None],
        ], dim=1)
        image = image.type(torch.int)

        # tile
        global_idx = map_meta[idx][1]
        city = map_meta[idx][0]
        tile_pathname = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
        tile = cv2.imread(tile_pathname)
        point_list = image.numpy()
        for point in point_list:
            cv2.circle(tile,point,1,(255,0,0),1)        
        tile_filename = f'{ndx}_map_{i}_{yaw}.png'
        cv2.imwrite(os.path.join(output_folder, tile_filename),tile)
        
        
        
    
        





