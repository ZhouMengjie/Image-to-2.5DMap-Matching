import os
import sys
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd
    
def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)       
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]
    return sample

def sample_bilinear(signal, rx, ry):
    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]
    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    bounds = (0, signal_dim_x, 0, signal_dim_y)
    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)
    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11
    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2

def get_polar(tile, height=7, width=14):
    S = 7
    i = np.arange(0, height)
    j = np.arange(0, width)
    jj, ii = np.meshgrid(j, i)
    y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
    x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)
    polar_tile = sample_bilinear(tile, x, y)
    return polar_tile

if __name__ == "__main__":
    dir_name = 'feature_maps/2d_pol'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vis_embedding = sio.loadmat(os.path.join(os.getcwd(),'Image-to-2-5DMap','results','vis_2d_pol.mat'))
    map = vis_embedding['map']
    pano = vis_embedding['pano']
    tile_embedding = vis_embedding['tile']

    # up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    # up_tile_embedding = up(torch.tensor(tile_embedding))
    # up_tile_embedding = up_tile_embedding.numpy()

    seed = 20
    map = map[seed,...]
    pano = pano[seed, ...]
    tile_embedding = tile_embedding[seed,...]

    labels = np.load(os.path.join(os.getcwd(),'Image-to-2-5DMap','results','labels.npy'))
    area = 'unionsquare5kU'
    data = pd.read_csv(os.path.join(os.getcwd(),'Image-to-2-5DMap','datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info = data.values   
    idx = labels[seed]
    global_idx = info[idx][1] # for train only
    city = info[idx][0]
    tile_path = os.path.join(os.getcwd(),'Image-to-2-5DMap','datasets','tiles_'+city+'_2019','z18',str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_path)
    tile.save(os.path.join(dir_name, str(idx)+'_map.png'))   

    data = pd.read_csv(os.path.join(os.getcwd(),'Image-to-2-5DMap','datasets', 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    panoid = info[idx][0]
    img_path = os.path.join(os.getcwd(),'Image-to-2-5DMap','datasets', 'jpegs_'+city+'_2019', panoid + '.jpg')
    img = Image.open(img_path)
    img.show()
    img.save(os.path.join(dir_name,str(idx)+'_pano.png'))

    # Save the whole feature map
    map = np.sum(map, axis=0)
    map_normalized_map = (map - np.min(map)) / (np.max(map) - np.min(map))
    plt.imshow(map_normalized_map, cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(dir_name,'map_'+str(seed)+'.png'), bbox_inches='tight')
    plt.close()

    # map to polar
    # map_normalized_map = np.expand_dims(map_normalized_map,2)
    # polar_map = get_polar(map_normalized_map)
    # polar_map_normalized_map = polar_map.squeeze(2)
    # plt.imshow(polar_map_normalized_map, cmap='jet')
    # plt.axis('off')
    # plt.savefig(os.path.join(dir_name,'map_pol_'+str(seed)+'.png'), bbox_inches='tight')
    # plt.close()

    pano = np.sum(pano, axis=0)
    pano_normalized_map = (pano - np.min(pano)) / (np.max(pano) - np.min(pano))    
    plt.imshow(pano_normalized_map, cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(dir_name,'pano_'+str(seed)+'.png'), bbox_inches='tight')
    plt.close()

    # tile embedding
    # tile_feature_map = np.sum(tile_embedding, axis=0)
    # tile_normalized_map = (tile_feature_map - np.min(tile_feature_map)) / (np.max(tile_feature_map) - np.min(tile_feature_map))
    # plt.imshow(tile_normalized_map, cmap='jet')
    # plt.axis('off')
    # plt.savefig(os.path.join(dir_name,'tile_'+str(seed)+'.png'), bbox_inches='tight')
    # plt.close()

    # # tile to polar
    # tile_normalized_map = np.expand_dims(tile_normalized_map,2)
    # polar_tile = get_polar(tile_normalized_map)
    # polar_tile_normalized_map = polar_tile.squeeze(2)
    # plt.imshow(polar_tile_normalized_map, cmap='jet')
    # plt.axis('off')
    # plt.savefig(os.path.join(dir_name,'tile_pol_'+str(seed)+'.png'), bbox_inches='tight')
    # plt.close()

    # apply feature map on original pano and polared_map
    image_array = np.array(img)
    pano_resized_map = np.array(Image.fromarray(pano_normalized_map).resize(image_array.shape[:2][::-1], Image.ANTIALIAS))
    plt.imshow(image_array)
    plt.imshow(pano_resized_map, cmap='jet', alpha=0.5)  # Adjust the alpha value to control the intensity of the heatmap overlay
    plt.colorbar()  # Add a colorbar to indicate the heatmap intensity
    plt.axis('off')
    plt.savefig(os.path.join(dir_name,'pano_heatmap_'+str(seed)+'.png'), bbox_inches='tight')
    plt.close()

    # img = Image.open(os.path.join(dir_name,str(seed)+'_polar.png'))
    # image_array = np.array(img)
    # map_resized_map = np.array(Image.fromarray(polar_map_normalized_map).resize(image_array.shape[:2][::-1], Image.ANTIALIAS))
    # plt.imshow(image_array)
    # plt.imshow(map_resized_map, cmap='jet', alpha=0.5)  # Adjust the alpha value to control the intensity of the heatmap overlay
    # plt.colorbar()  # Add a colorbar to indicate the heatmap intensity
    # plt.axis('off')
    # plt.savefig(os.path.join(dir_name,'map_heatmap_'+str(seed)+'.png'), bbox_inches='tight')
    # plt.close()

    img = Image.open(os.path.join(dir_name,str(seed)+'_polar.png'))
    image_array = np.array(img)
    map_resized_map = np.array(Image.fromarray(map_normalized_map).resize(image_array.shape[:2][::-1], Image.ANTIALIAS))
    plt.imshow(image_array)
    plt.imshow(map_resized_map, cmap='jet', alpha=0.5)  # Adjust the alpha value to control the intensity of the heatmap overlay
    plt.colorbar()  # Add a colorbar to indicate the heatmap intensity
    plt.axis('off')
    plt.savefig(os.path.join(dir_name,'map_heatmap_'+str(seed)+'.png'), bbox_inches='tight')
    plt.close()

    # tile_resized_map = np.array(Image.fromarray(polar_tile_normalized_map).resize(image_array.shape[:2][::-1], Image.ANTIALIAS))
    # plt.imshow(image_array)
    # plt.imshow(tile_resized_map, cmap='jet', alpha=0.5)  # Adjust the alpha value to control the intensity of the heatmap overlay
    # plt.colorbar()  # Add a colorbar to indicate the heatmap intensity
    # plt.axis('off')
    # plt.savefig(os.path.join(dir_name,'tile_heatmap_'+str(seed)+'.png'), bbox_inches='tight')
    # plt.close()









    






