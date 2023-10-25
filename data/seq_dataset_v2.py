import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import torch
import cv2
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from PIL import Image
from data.augmentation_simple import TrainTransform, ValTransform, TrainRGBTransform, ValRGBTransform, TrainTileTransform, ValTileTransform


class SeqDataset_v2(Dataset):
    def __init__(self, dataset_path: str, query_filename: str,
                image_size=224, tile_size=224,
                use_polar: bool=False, use_cloud: bool=False,
                normalize: bool=True, npoints=1024,
                image_transform=None, tile_transform=None, cloud_transform=None):
        self.dataset_path = dataset_path
        self.seq_filepath = os.path.join(dataset_path, 'csv', query_filename+'_sq.csv')
        self.sequences = (pd.read_csv(self.seq_filepath, sep=',', header=None)).values
        self.indexes = np.arange(0, len(self.sequences))
        self.area = query_filename+'_idx'

        self.pano_filepath = os.path.join(dataset_path, 'csv', query_filename+'.pickle')
        self.pano: Dict[int, TrainingTuple] = pickle.load(open(self.pano_filepath, 'rb')) 
        self.map_filepath = os.path.join(dataset_path, 'csv', query_filename+ '_set.csv')
        self.map = (pd.read_csv(self.map_filepath, sep=',', header=None)).values
        points_mh = np.load(os.path.join(dataset_path, 'manhattan', 'manhattanU.npy'))
        points_pt = np.load(os.path.join(dataset_path, 'pittsburgh', 'pittsburghU.npy'))
        self.points = {'manhattan': points_mh, 'pittsburgh': points_pt}

        self.image_size = image_size
        self.tile_size = tile_size
        self.use_polar = use_polar
        self.use_cloud = use_cloud
        self.normalize = normalize
        self.npoints = npoints

        self.image_transform = image_transform
        self.tile_transform = tile_transform
        self.cloud_transform = cloud_transform

        print('{} queries in the dataset'.format(len(self)))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, ndx):
        # Load single panos, tiles, point clouds for each sequence
        pano_seq = []
        tile_seq = []
        cloud_seq = [] 
        center_seq = []
        xyz_seq = []
        seq_indices = self.sequences[ndx]
        for idx in seq_indices:
            panoid = self.pano[idx].panoid # panoid
            city = self.pano[idx].city
            global_idx = self.map[idx][1]

            pano_pathname = os.path.join(self.dataset_path, 'jpegs_'+city+'_2019', panoid+'.jpg')
            img = cv2.imread(pano_pathname)
            if self.image_size is not None:
                img = cv2.resize(img, (self.image_size*2, self.image_size))
            img = self.image_transform(img)
            pano_seq.append(img) # C, H, W

            tile_pathname = os.path.join(self.dataset_path, 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
            tile = Image.open(tile_pathname).convert('RGB')
            tile = self.tile_transform(tile) # C, H, W
            if self.use_polar:
                tile = tile.permute(2,1,0).permute(1,0,2).numpy() # h, w, c
                polar_tile = self.get_polar(tile) # h, w, c
                tile = torch.tensor(polar_tile, dtype=torch.float).permute(2,1,0).permute(0,2,1) # c, h, w
            tile_seq.append(tile)

            if self.use_cloud:
                center_x = self.pano[idx].center_x
                center_y = self.pano[idx].center_y
                heading = self.pano[idx].heading # in degree
                pcd = {'center': [center_x, 0, center_y],'heading': heading}
                pcd_folder = self.area.split('_')
                pcd_pathname = os.path.join(self.dataset_path, pcd_folder[0]+'_idx', panoid+'.npy')
                if not os.path.exists(pcd_pathname):
                    pcd_pathname = os.path.join(self.dataset_path, pcd_folder[1]+'_idx', panoid+'.npy')
                # Load point cloud and apply transform
                coords = self.load_pc(pcd_pathname, self.points[city])
                pcd['cloud'] = coords
                pcd['cloud_ft'] = coords # useless
                pcd = self.cloud_transform(pcd)
                if self.npoints is not None:
                    pc = pcd['cloud'].numpy()
                    coords = self.farthest_point_sample(pc, self.npoints)
                    pcd['xyz'] = torch.tensor(coords, dtype=torch.float)
                    pcd['cloud'] = torch.tensor(coords, dtype=torch.float)
                if self.normalize:
                    pc = pcd['cloud'].numpy()
                    coords = self.pc_normalize(pc)
                    pcd['cloud'] = torch.tensor(coords, dtype=torch.float)
                cloud_seq.append(pcd['cloud'])
                xyz_seq.append(pcd['xyz'])
                center_seq.append(torch.tensor(pcd['center']))
            
        pano_seq = torch.stack(pano_seq, dim=0) # T, C, H, W
        tile_seq = torch.stack(tile_seq, dim=0) # T, C, H, W
        cloud_seq = torch.stack(cloud_seq, dim=0) # T, N, C
        xyz_seq = torch.stack(xyz_seq, dim=0) # T, N, 3
        center_seq = torch.stack(center_seq, dim=0) # T, 3
        return {'images':pano_seq, 'tiles':tile_seq, 'coords':cloud_seq, 'xyz':xyz_seq,
                'center':center_seq,'label':torch.tensor(ndx)}

    def load_pc(self, filename, points):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        vertex_indices = np.load(filename)
        coords = points[vertex_indices][:]
        coords = torch.tensor(coords, dtype=torch.float)
        return coords

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def farthest_point_sample(self, point, npoint):
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
        return pt 

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


def make_collate_fn(dataset: SeqDataset_v2):
    def collate_fn(data_list):
        result = {}
        labels = [d['label'] for d in data_list]
        result['label'] = torch.tensor(labels)

        center = torch.cat([d['center'] for d in data_list])
        result['center'] = center

        images_batch = torch.cat([d["images"] for d in data_list])
        result['images'] = images_batch

        tiles_batch = torch.cat([d["tiles"] for d in data_list])
        result['tiles'] = tiles_batch

        if dataset.use_cloud:
            coordinates_batch = torch.cat([d["coords"] for d in data_list])
            xyz_batch = torch.cat([d["xyz"] for d in data_list])
            result['coords'] = coordinates_batch.permute(0, 2, 1)
            result['xyz'] = xyz_batch.permute(0, 2, 1)

        return result
    return collate_fn

if __name__ == '__main__':
    dataset_path = 'datasets'
    query_filename = 'unionsquare5kU'
    aug_mode = 0
    tile_size = 224
    dataset = {}
    image_train_transform = TrainRGBTransform(aug_mode)
    tile_train_transform = TrainTileTransform(aug_mode, tile_size)
    cloud_train_transform = TrainTransform(aug_mode)

    dataset['train'] = SeqDataset_v2(dataset_path, query_filename, use_cloud=True, 
                    image_transform=image_train_transform, tile_transform=tile_train_transform,
                    cloud_transform=cloud_train_transform)

    collate_fn = make_collate_fn(dataset['train'])
    
    batch_sampler = None
    batch_size = 4
    nw = 0
    device = 'cuda'
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset['train'], batch_sampler=batch_sampler, batch_size=batch_size, collate_fn=collate_fn,
                            num_workers=nw, pin_memory=True, shuffle=True, drop_last=True)
    
    for batch in dataloaders['train']:
        batch = {e: batch[e].to(device) for e in batch}
        print(batch['images'].shape)
        print(batch['tiles'].shape)
        print(batch['coords'].shape)
        print(batch['xyz'].shape)
        print(batch['center'].shape)
        print(batch['label'].shape)



        



   

    
   

    
    
    
        

