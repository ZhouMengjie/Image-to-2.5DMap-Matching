import os
import sys
sys.path.append(os.getcwd())
import yaml
import numpy as np
import pandas as pd
import cv2
import torch
from data import augmentation
import torchvision.transforms as transforms
import scipy.io as sio

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
        # dist = np.sum((xyz - centroid) ** 2, -1)
        dist = -2 * np.matmul(xyz, centroid)
        dist += xyz2
        dist +=  np.sum(centroid ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pt = point[centroids.astype(np.int32)]
    ft = feature[centroids.astype(np.int32)]
    return pt, ft


if __name__ == "__main__":
    # load panorama 
    data_path = 'datasets'
    city = 'manhattan'
    area = 'unionsquare5kU'
    save_for_vis = sio.loadmat('us_vis.mat')
    key_frame = save_for_vis['save_for_vis'][2] #0-gt, 1-es, 2-mes
    
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info1 = data.values
    label = pd.read_csv(os.path.join(data_path, 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info2 = label.values

    # save tile for animation
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    classes = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = classes.values
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)       
    # partition
    for idx in key_frame:
        panoid = info1[idx][0]
        center_x = info1[idx][1]
        center_y = info1[idx][2]
        heading = info1[idx][3]
        vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
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
        t = [augmentation.RandomRotation(max_theta=0, axis=np.asarray([0,1,0])),
            augmentation.RandomCenterCrop(radius=76, rnd=0)]
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
        global_idx = info2[idx][1] # for train only
        city = info2[idx][0]
        tile_pathname = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
        tile = cv2.imread(tile_pathname)
        point_list = image.numpy()
        for point in point_list:
            cv2.circle(tile,point,1,(255,0,0),1)
        cv2.imwrite(os.path.join('video',area,str(idx)+'_mes.jpg'),tile)



    






    




