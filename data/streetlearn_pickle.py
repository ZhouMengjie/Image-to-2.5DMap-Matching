import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from typing import Dict
from data.Equirec2Perspec import Equirectangular
import torchvision.transforms as transforms
import io
import copy
import cv2
import matplotlib.pyplot as plt

def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)

class StreetLearnDataset(Dataset):
    def __init__(self, dataset_path: str, query_filename: str, 
                transform=None, 
                image_size=None, image_transform=None,
                tile_size=None, tile_transform=None,
                use_cloud: bool = True, use_rgb: bool = True, use_tile: bool = False,
                use_feat: bool = True, use_polar = False,
                normalize: bool = False, npoints=8192):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.area = query_filename+'_idx'
        self.query_filepath = os.path.join(dataset_path, 'csv', query_filename+'.pickle')
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb')) 
        self.set_filepath = os.path.join(dataset_path, 'csv', query_filename+ '_set.csv')
        self.global_queries= (pd.read_csv(self.set_filepath, sep=',', header=None)).values
        self.transform = transform
        self.indexes = np.arange(0, len(self.queries))
        self.image_size = image_size
        self.image_transform = image_transform
        self.tile_size = tile_size
        self.tile_transform = tile_transform
        self.use_cloud = use_cloud
        self.use_rgb = use_rgb
        self.use_tile = use_tile
        self.use_feat = use_feat
        self.use_polar = use_polar
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
        self.initialized = False                
        print('{} queries in the dataset'.format(len(self)))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        panoid = self.queries[ndx].panoid # panoid
        center_x = self.queries[ndx].center_x
        center_y = self.queries[ndx].center_y
        heading = self.queries[ndx].heading # in degree
        city = self.queries[ndx].city
        global_idx = self.global_queries[ndx][1]
        assert self.global_queries[ndx][0] == city
        result = {'ndx': ndx, 'center': [center_x, 0, center_y], 
                  'heading': heading, 'city': city}
        result_aug = copy.deepcopy(result)
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
            if self.transform.aug_mode == 1:
                result_aug = copy.deepcopy(result)
            # -----------transform 1---------- #
            result = self.transform(result)
            if self.npoints is not None:
                pc = result['cloud'].numpy()
                feat = result['cloud_ft'].numpy()
                # filter = self.filter_semantic(feat)
                # if filter:
                #     pc = pc[filter]
                #     feat = feat[filter]
                # else:
                #     print(pcd_folder[0]+':'+str(ndx))
                coords, feats = self.farthest_point_sample(pc, feat, self.npoints)
                # coords, feats = self.random_point_sample(pc, feat, self.npoints)
                result['xyz'] = torch.tensor(coords, dtype=torch.float)
                result['cloud'] = torch.tensor(coords, dtype=torch.float)
                result['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
            if self.normalize:
                pc = result['cloud'].numpy()
                coords = self.pc_normalize(pc)
                result['cloud'] = torch.tensor(coords, dtype=torch.float)
            if self.use_feat:
                result['cloud'] = torch.cat((result['cloud'],result['cloud_ft']),-1)
            # -----------transform 2---------- #
            if self.transform.aug_mode == 1:
                result_aug = self.transform(result_aug)
                if self.npoints is not None:
                    pc = result_aug['cloud'].numpy()
                    feat = result_aug['cloud_ft'].numpy()
                    coords, feats = self.farthest_point_sample(pc, feat, self.npoints)
                    result_aug['xyz'] = torch.tensor(coords, dtype=torch.float)
                    result_aug['cloud'] = torch.tensor(coords, dtype=torch.float)
                    result_aug['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
                if self.normalize:
                    pc = result_aug['cloud'].numpy()
                    coords = self.pc_normalize(pc)
                    result_aug['cloud'] = torch.tensor(coords, dtype=torch.float)
                if self.use_feat:
                    result_aug['cloud'] = torch.cat((result_aug['cloud'],result_aug['cloud_ft']),-1)
           
        if self.use_rgb:
            pano_pathname = os.path.join(self.dataset_path, 'jpegs_'+city+'_2019', panoid+'.jpg')
            assert os.path.isfile(pano_pathname), "Image {} not found".format(pano_pathname)
            img = cv2.imread(pano_pathname)
            if self.image_size is not None:
                img = cv2.resize(img, (self.image_size*2, self.image_size))
            if self.image_transform.aug_mode == 1:
                img_aug = copy.deepcopy(img)
            # -----------transform 1---------- #
            img = self.image_transform(img)
            result['image'] = img # C, H, W
            # -----------transform 2---------- #
            if self.image_transform.aug_mode == 1:
                img_aug = self.image_transform(img_aug)
                result_aug['image'] = img_aug

        if self.use_tile:
            tile_pathname = os.path.join(self.dataset_path, 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
            assert os.path.isfile(tile_pathname), "Image {} not found".format(tile_pathname)
            tile = Image.open(tile_pathname).convert('RGB')
            if self.tile_transform.aug_mode == 1:
                tile_aug = copy.deepcopy(tile)
            # -----------transform 1---------- #
            tile = self.tile_transform(tile) # C, H, W
            if self.use_polar:
                tile = tile.permute(2,1,0).permute(1,0,2).numpy() # h, w, c
                polar_tile = self.get_polar(tile) # h, w, c
                tile = torch.tensor(polar_tile, dtype=torch.float).permute(2,1,0).permute(0,2,1) # c, h, w
            result['tile'] = tile
            # -----------transform 2---------- #
            if self.tile_transform.aug_mode == 1:
                tile_aug = self.tile_transform(tile_aug)
                if self.use_polar:
                    tile_aug = tile_aug.permute(2,1,0).permute(1,0,2).numpy()
                    polar_tile = self.get_polar(tile_aug)
                    tile_aug = torch.tensor(polar_tile, dtype=torch.float).permute(2,1,0).permute(0,2,1)
                result_aug['tile'] = tile_aug

        return result, result_aug

    def get_postitives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

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

    def random_point_sample(self, point, feature, npoint):
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

    def filter_semantic(self, array):
        index = []
        for row_index, row in enumerate(array):
            if row[4] == 1 or row[19] == 1 or row[13] == 1:
                index.append(row_index)
        return index

    def get_snaps(self, pano, fov=90, size=224):
        snaps = []
        equ = Equirectangular(pano)           
        views = [0,-90,90,180]            
        H, W = size if hasattr(size,'__iter__') else (size,size) 
        snaps = [equ.GetPerspective(fov, t, 0, H, W) for t in views]
        return snaps

    def get_polar(self, tile, height=224, width=448):
        S = self.tile_size
        i = np.arange(0, height)
        j = np.arange(0, width)
        jj, ii = np.meshgrid(j, i)
        y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
        x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)
        polar_tile = self.sample_bilinear(tile, x, y)
        return polar_tile
    
    def sample_within_bounds(self, signal, x, y, bounds):
        xmin, xmax, ymin, ymax = bounds
        idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)       
        sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
        sample[idxs, :] = signal[x[idxs], y[idxs], :]
        return sample

    def sample_bilinear(self, signal, rx, ry):
        signal_dim_x = signal.shape[0]
        signal_dim_y = signal.shape[1]
        # obtain four sample coordinates
        ix0 = rx.astype(int)
        iy0 = ry.astype(int)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        bounds = (0, signal_dim_x, 0, signal_dim_y)
        # sample signal at each four positions
        signal_00 = self.sample_within_bounds(signal, ix0, iy0, bounds)
        signal_10 = self.sample_within_bounds(signal, ix1, iy0, bounds)
        signal_01 = self.sample_within_bounds(signal, ix0, iy1, bounds)
        signal_11 = self.sample_within_bounds(signal, ix1, iy1, bounds)
        na = np.newaxis
        # linear interpolation in x-direction
        fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
        fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11
        # linear interpolation in y-direction
        return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2
        
    
class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: str, center_x: float, center_y: float, heading: float,
                city: str, positives: np.ndarray, non_negatives: np.ndarray):
        self.panoid = id
        self.center_x = center_x
        self.center_y = center_y
        self.heading = heading
        self.city = city
        self.positives = positives
        self.non_negatives = non_negatives

