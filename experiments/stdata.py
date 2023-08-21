# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import  OneHotEncoder

DEBUG = False

class STData:
    def __init__(self, dataset_path: str, query_filename: str, 
                transform=None, set_transform=None, image_size=None, image_transform=None,
                use_cloud: bool = True, use_rgb: bool = True, use_feat: bool = True, 
                normalize: bool = True, npoints=8192):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.area = query_filename+'_idx'
        self.query_filepath = os.path.join(dataset_path, 'csv', query_filename+'_xy.csv')
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries = (pd.read_csv(self.query_filepath, sep=',', header=None)).values
        # db = pd.read_csv(self.query_filepath, sep=',', header=None)
        # self.queries = db.sample(n=5000, random_state=10).values
        self.indexes = np.arange(0, len(self.queries))
        self.image_size = image_size
        self.image_transform = image_transform
        self.use_cloud = use_cloud
        self.use_rgb = use_rgb
        self.use_feat = use_feat
        self.normalize = normalize
        self.npoints = npoints
        points_mh = np.load(os.path.join(dataset_path, 'manhattan', 'manhattanU.npy'))
        points_pt = np.load(os.path.join(dataset_path, 'pittsburgh', 'pittsburghU.npy'))
        self.points = {'manhattan': points_mh, 'pittsburgh': points_pt}
        # encode semantic features
        features_mh = (pd.read_csv(os.path.join(dataset_path, 'manhattan', 'manhattanU.csv'), sep=',', header=None)).values
        features_pt = (pd.read_csv(os.path.join(dataset_path, 'pittsburgh', 'pittsburghU.csv'), sep=',', header=None)).values
        encoder = OneHotEncoder()
        encoder.fit(np.concatenate((features_mh, features_pt), axis=0))
        features_mh = encoder.transform(features_mh).toarray()
        features_pt = encoder.transform(features_pt).toarray()
        self.features = {'manhattan': features_mh, 'pittsburgh': features_pt}             

    def __call__(self, ndx):
        # Load point cloud and apply transform
        panoid = self.queries[ndx][0] # panoid
        center_x = self.queries[ndx][1]
        center_y = self.queries[ndx][2]
        heading = self.queries[ndx][3] # in degree
        city = self.queries[ndx][4]
        result = {'ndx': ndx, 'center': [center_x, 0, center_y], 
                  'heading': heading, 'city': city}
        if self.use_cloud:
            pcd_folder = self.area.split('_')
            pcd_pathname = os.path.join(self.dataset_path, pcd_folder[0]+'_idx', panoid+'.npy')
            if not os.path.exists(pcd_pathname):
                pcd_pathname = os.path.join(self.dataset_path, pcd_folder[1]+'_idx', panoid+'.npy')
                assert os.path.isfile(pcd_pathname), "Map {} not found".format(pcd_pathname)
            # Load point cloud and apply transform
            coords, feats = self.load_pc(pcd_pathname, self.points[city], self.features[city])
            result['cloud'] = coords
            result['cloud_ft'] = feats
            if self.transform is not None:
                result = self.transform(result)
            if self.npoints is not None:
                pc = result['cloud'].numpy()
                feat = result['cloud_ft'].numpy()
                coords, feats = self.farthest_point_sample(pc, feat, self.npoints)
                result['cloud'] = torch.tensor(coords, dtype=torch.float)
                result['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
            if self.normalize:
                pc = result['cloud'].numpy()
                coords = self.pc_normalize(pc)
                result['cloud'] = torch.tensor(coords, dtype=torch.float)
            if self.use_feat:
                result['cloud'] = torch.cat((result['cloud'],result['cloud_ft']),-1)
            
        if self.use_rgb:
            pano_pathname = os.path.join(self.dataset_path, 'jpegs_'+city+'_2019', panoid+'.jpg')
            assert os.path.isfile(pano_pathname), "Image {} not found".format(pano_pathname)
            img = Image.open(pano_pathname)
            # transform
            if self.image_size is not None:
                img = img.resize([self.image_size*2, self.image_size])
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img

        return result

    def load_pc(self, filename, points, features):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        vertex_indices = np.load(filename)
        coords = points[vertex_indices][:]
        feats = features[vertex_indices]
        coords = torch.tensor(coords, dtype=torch.float)
        feats = torch.tensor(feats, dtype=torch.float)
        return coords, feats

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def farthest_point_sample(self, point, feature, npoint):
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
        farthest = np.random.randint(0, N)
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
        # sample or repeat here
        if N >= npoint:
            pt = point[centroids.astype(np.int32)]
            ft = feature[centroids.astype(np.int32)]
        else: # repeat
            pt = [point[centroid.astype(np.int32)] for centroid in centroids]
            ft = [feature[centroid.astype(np.int32)] for centroid in centroids]
        return pt, ft