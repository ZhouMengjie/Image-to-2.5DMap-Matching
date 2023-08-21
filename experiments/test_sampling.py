import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder
from experiments.visualizer import visualize_pcd  

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

def random_point_sample(point, feature, npoint):
    N, D = point.shape
    pt_idxs = np.arange(0, N)
    np.random.shuffle(pt_idxs)
    if N >= npoint:
        pt_idxs = pt_idxs[0:npoint]
    else:
        pt_idxs = np.random.choice(pt_idxs, size=npoint, replace=True)
    pt = point[pt_idxs]
    ft = feature[pt_idxs]
    return pt, ft   

def filter_semantic(array, column):
        index = []
        for row_index, row in enumerate(array):
            if row[column] == 1:
                index.append(row_index)
        return index

if __name__ == "__main__":
        dataset_path = 'datasets'
        query_filename = 'hudsonriver5kU'
        area = query_filename+'_idx'
        query_filepath = os.path.join(dataset_path, 'csv', query_filename+'_xy.csv')   
        queries = (pd.read_csv(query_filepath, sep=',', header=None)).values
        npoints = 4096
        points_mh = np.load(os.path.join(dataset_path, 'manhattan', 'manhattanU.npy'))
        points_pt = np.load(os.path.join(dataset_path, 'pittsburgh', 'pittsburghU.npy'))
        points = {'manhattan': points_mh, 'pittsburgh': points_pt}
        # encode semantic features
        features_mh = (pd.read_csv(os.path.join(dataset_path, 'manhattan', 'manhattanU.csv'), sep=',', header=None)).values
        features_pt = (pd.read_csv(os.path.join(dataset_path, 'pittsburgh', 'pittsburghU.csv'), sep=',', header=None)).values
        encoder = OneHotEncoder()
        encoder.fit(np.concatenate((features_mh, features_pt), axis=0))
        features_mh = encoder.transform(features_mh).toarray()
        features_pt = encoder.transform(features_pt).toarray()
        features = {'manhattan': features_mh, 'pittsburgh': features_pt}

        ndx = 25
        panoid = queries[ndx][0] # panoid
        center_x = queries[ndx][1]
        center_y = queries[ndx][2]
        heading = queries[ndx][3] # in degree
        city = queries[ndx][4]
        filename = os.path.join(dataset_path, query_filename+'_idx', panoid+'.npy')
        vertex_indices = np.load(filename)
        pc = points[city][vertex_indices][:]
        feat = features[city][vertex_indices]
        filter = filter_semantic(feat, column=4)
        if filter:
            pc = pc[filter]
            feat = feat[filter]
        else:
            print(query_filename+':'+str(ndx))
        coords, feats = farthest_point_sample(pc, feat, npoints)
        # coords, feats = random_point_sample(pc, feat, npoints)

        with open('color_map.yaml','r') as f:
            colors = yaml.load(f, Loader=yaml.FullLoader)
        color = []
        for i in range(len(feats)):
            class_id = np.argmax(feats[i])
            color.append(np.divide(colors['color_map'][class_id], 255))
        coord = np.asarray(coords)
        color = np.asarray(color)
        visualize_pcd(coord, color, 'npy')




