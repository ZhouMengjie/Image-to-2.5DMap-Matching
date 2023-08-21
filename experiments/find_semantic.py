import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
# import open3d as o3d
import yaml

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
    # load index - top5
    area = 'wallstreet5kU'
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info1 = data.values
    label = pd.read_csv(os.path.join('datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info2 = label.values
   
    city = 'manhattan'
    # load all points and semantic labels
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    classes = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = classes.values
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)
       
    # partition
    water = []
    for i in range(5000):
        number = 0
        panoid = info1[i][0]
        center_x = info1[i][1]
        center_y = info1[i][2]
        heading = info1[i][3]
        vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
        coords = points[vertex_indices][:]
        feats = semantic_ids[vertex_indices]
        colors = []
        for j in range(len(feats)):
            class_id = feats[j][0]
            if class_id == 19:
                number += 1
                if number >= 300:
                    water.append(i)
                    break
    print(water)
                


   