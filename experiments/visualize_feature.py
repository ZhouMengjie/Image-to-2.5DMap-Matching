import os
import sys
sys.path.append(os.getcwd())
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from experiments.visualizer import visualize_pcd
import pandas as pd
from sklearn.decomposition import PCA

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def flatten(*Fs, masks=None):
    if masks is not None:
        assert len(Fs) == len(masks)

    flatten = []
    for i, F in enumerate(Fs):
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F_flat = F.reshape(-1, c)
        if masks is not None and masks[i] is not None:
            mask = masks[i]
            assert mask.shape == F.shape[:2]
            F_flat = F_flat[mask.reshape(-1)]
        flatten.append(F_flat)
    flatten = np.concatenate(flatten, axis=0) # h*w, c
    flatten = normalize(flatten) # h*w, c
    return flatten

def rgb(*Fs, flatten, masks=None):
    flatten = normalize(flatten) # h*w, c
    Fs_rgb = []
    for i, F in enumerate(Fs):
        h, w = F.shape[-2:]
        if masks is None or masks[i] is None:
            F_rgb, flatten = np.split(flatten, [h * w], axis=0)
            F_rgb = F_rgb.reshape((h, w, 3))
        else:
            F_rgb = np.zeros((h, w, 3))
            indices = np.where(masks[i])
            F_rgb[indices], flatten = np.split(flatten, [len(indices[0])], axis=0)
            F_rgb = np.concatenate([F_rgb, masks[i][..., None]], axis=-1)
        Fs_rgb.append(F_rgb)
    assert flatten.shape[0] == 0, flatten.shape
    return Fs_rgb

def features_to_RGB(*Fs, masks=None, skip=1):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""

    if masks is not None:
        assert len(Fs) == len(masks)

    flatten = []
    for i, F in enumerate(Fs):
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F_flat = F.reshape(-1, c)
        if masks is not None and masks[i] is not None:
            mask = masks[i]
            assert mask.shape == F.shape[:2]
            F_flat = F_flat[mask.reshape(-1)]
        flatten.append(F_flat)
    flatten = np.concatenate(flatten, axis=0) # h*w, c
    flatten = normalize(flatten) # h*w, c

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(flatten[::skip])
        flatten = pca.transform(flatten)
    else:
        flatten = pca.fit_transform(flatten)
    flatten = (normalize(flatten) + 1) / 2

    Fs_rgb = []
    for i, F in enumerate(Fs):
        h, w = F.shape[-2:]
        if masks is None or masks[i] is None:
            F_rgb, flatten = np.split(flatten, [h * w], axis=0)
            F_rgb = F_rgb.reshape((h, w, 3))
        else:
            F_rgb = np.zeros((h, w, 3))
            indices = np.where(masks[i])
            F_rgb[indices], flatten = np.split(flatten, [len(indices[0])], axis=0)
            F_rgb = np.concatenate([F_rgb, masks[i][..., None]], axis=-1)
        Fs_rgb.append(F_rgb)
    assert flatten.shape[0] == 0, flatten.shape
    return Fs_rgb
    
if __name__ == "__main__":
    pano_batch = np.load(os.path.join('results', 'feature_maps', 'pano3.npy')) # b, c, h, w
    tile_batch = np.load(os.path.join('results', 'feature_maps', 'tile3.npy')) # b, c, h, w
    # tile3d_batch = np.load(os.path.join('results', 'feature_maps', 'tile3d.npy')) # b, c, h, w
    # fuse_batch = np.load(os.path.join('results', 'feature_maps', 'fuse2.npy')) # b, c, h, w


    idx = 20
    pano = pano_batch[idx,...]
    tile = tile_batch[idx,...]
    # tile3d = tile3d_batch[idx,...]
    # fuse = fuse_batch[idx,...]

    # display the whole feature map
    fmap_rgb = features_to_RGB(pano)
    fmap_rgb = fmap_rgb[0]
    # plt.figure(dpi=3,figsize=(448,224))
    plt.imshow(fmap_rgb,cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join('results','feature_maps','pano3.png'), bbox_inches='tight')
    plt.close()

    fmap_rgb = features_to_RGB(tile)
    fmap_rgb = fmap_rgb[0]
    # plt.figure(dpi=3,figsize=(224,224))
    plt.imshow(fmap_rgb,cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join('results','feature_maps','tile3.png'), bbox_inches='tight')
    plt.close()

    # tile_tensor = torch.tensor(tile_batch,dtype=float)
    # up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    # tile_up = up(tile_tensor)[idx,...]
    # fmap_rgb = features_to_RGB(tile_up.numpy())
    # fmap_rgb = fmap_rgb[0]
    # # plt.figure(dpi=3,figsize=(224,224))
    # plt.imshow(fmap_rgb,cmap='jet')
    # plt.axis('off')
    # plt.savefig(os.path.join('results','feature_maps','tileup.png'), bbox_inches='tight')
    # plt.close()

    # fmap_rgb = features_to_RGB(tile3d)
    # fmap_rgb = fmap_rgb[0]
    # # plt.figure(dpi=3,figsize=(224,224))
    # plt.imshow(fmap_rgb,cmap='jet')
    # plt.axis('off')
    # plt.savefig(os.path.join('results','feature_maps','tile3d.png'), bbox_inches='tight')
    # plt.close()

    # fmap_rgb = features_to_RGB(fuse)
    # fmap_rgb = fmap_rgb[0]
    # # plt.figure(dpi=3,figsize=(224,224))
    # plt.imshow(fmap_rgb,cmap='jet')
    # plt.axis('off')
    # plt.savefig(os.path.join('results','feature_maps','fuse2.png'), bbox_inches='tight')
    # plt.close()


    # # display the point cloud
    xyz = np.load(os.path.join('results', 'feature_maps', 'xyz3.npy')) # b, c, n
    xyz = xyz[idx,...]
    xyz = xyz.transpose(1,0)

    point = np.load(os.path.join('results', 'feature_maps', 'point3.npy')) # b, c, n
    point = point[idx, ...]
    point = point.transpose(1,0) # n, c
    pca = PCA(n_components=3)
    point = pca.fit_transform(point)
    point = (normalize(point) + 1) / 2
    visualize_pcd(xyz, point, 'npy')

    # point = np.load(os.path.join('results', 'feature_maps', 'point2d.npy')) # b, c, n
    # point = point[idx, ...]
    # point = point.transpose(1,0) # n, c
    # pca = PCA(n_components=3)
    # point = pca.fit_transform(point)
    # point = (normalize(point) + 1) / 2
    # visualize_pcd(xyz, point, 'npy')

    # fuse = np.load(os.path.join('results', 'feature_maps', 'fuse.npy')) # b, c, n
    # fuse = fuse[idx, ...]
    # fuse = fuse.transpose(1,0) # n, c
    # pca = PCA(n_components=3)
    # fuse = pca.fit_transform(fuse)
    # fuse = (normalize(fuse) + 1) / 2
    # visualize_pcd(xyz, fuse, 'npy')











    






