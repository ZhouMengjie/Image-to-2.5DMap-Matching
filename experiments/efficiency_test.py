import numpy as np
import pandas as pd
import time
import os
import open3d as o3d
import torch

def pc_normalize(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
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
    farthest = np.random.randint(0, N)
    xyz2 = np.sum(xyz ** 2, -1)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        # dist1 = np.sum((xyz - centroid) ** 2, -1)
        dist = -2 * np.matmul(xyz, centroid)
        dist += xyz2
        dist +=  np.sum(centroid ** 2, -1)
        # print(dist)
        # print(dist1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # sample or repeat here
    if N >= npoint:
        pt = point[centroids.astype(np.int32)]
        ft = feature[centroids.astype(np.int32)]
    else: # repeat
        pt = np.array([point[centroid.astype(np.int32)] for centroid in centroids])
        ft = np.array([feature[centroid.astype(np.int32)] for centroid in centroids])
        pt1 = point[centroids.astype(np.int32)]
        ft1 = feature[centroids.astype(np.int32)]
        # pt = [point[centroid.astype(np.int32)] for centroid in centroids]
        # ft = [feature[centroid.astype(np.int32)] for centroid in centroids]
    return pt, ft

def random_point_sample(point, feature, npoint):
    N, D = point.shape
    pt_idxs = np.arange(0, N)
    np.random.shuffle(pt_idxs)
    if N >= npoint:
        pt = point[pt_idxs[0:npoint]]
        ft = feature[pt_idxs[0:npoint]]
    else:
        pt_idxs = np.random.choice(pt_idxs, size=npoint, replace=True)
        pt = point[pt_idxs]
        ft = feature[pt_idxs]
        # pt = np.array([point[idx] for idx in pt_idxs])
        # ft = np.array([feature[idx] for idx in pt_idxs]) 
    return pt, ft 

if __name__ == '__main__':
    # np.random.seed(1)
    # point = np.random.randn(10000,3)
    # feature = np.random.rand(10000,24)
    # npoint = 8192

    torch.manual_seed(1)
    np.random.seed(1)

    point = torch.randn(3000,3)
    feature = torch.rand(3000,24)
    npoint = 4096

    start = time.time()
    point = point.numpy()
    feature = feature.numpy()
    pt, ft = farthest_point_sample(point, feature, npoint)
    # pt, ft = random_point_sample(point, feature, npoint)
    # pt = pc_normalize(point)
    point = torch.tensor(pt, dtype=torch.float)
    feature = torch.tensor(ft, dtype=torch.float)
    
    end = time.time()
    print('CPU runing time: ',end - start)
    print(len(pt))
