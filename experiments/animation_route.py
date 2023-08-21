import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import cv2
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
    gt_frame = save_for_vis['save_for_vis'][0]
    es_frame = save_for_vis['save_for_vis'][1]
    mes_frame = save_for_vis['save_for_vis'][2]
   
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info1 = data.values
    label = pd.read_csv(os.path.join(data_path, 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info2 = label.values

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowrite = cv2.VideoWriter(area+'pano.avi',fourcc,1,(1664,832))

    pano_array = []
    for idx in gt_frame:
        panoid = info1[idx][0]
        pano_path = os.path.join(data_path, ('jpegs_' + city + '_2019'), panoid + '.jpg')
        pano = cv2.imread(pano_path)
        pano_array.append(pano)

    for i in range(len(pano_array)):
        videowrite.write(pano_array[i])
    videowrite.release()

    # save ground truth tile
    videowrite = cv2.VideoWriter(area+'gt.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in gt_frame:
        tile_path = os.path.join('video', area, str(idx)+'_gt.jpg')
        # global_idx = info2[idx][1] 
        # tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

    # save es tile
    videowrite = cv2.VideoWriter(area+'es.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in es_frame:
        tile_path = os.path.join('video', area, str(idx)+'_es.jpg')
        # global_idx = info2[idx][1] 
        # tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

    # save mes tile
    videowrite = cv2.VideoWriter(area+'mes.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in mes_frame:
        tile_path = os.path.join('video', area, str(idx)+'_mes.jpg')
        # global_idx = info2[idx][1] 
        # tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()
    



    






    




