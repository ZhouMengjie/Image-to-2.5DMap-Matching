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
    # key_frame = [57,2110,2654,3478,3904,4016,4155,4207,4809,4980] # ws
    key_frame = [665,1306,1663,1865,3439,3717,3908,4077,4301,4348]  # us
  
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info1 = data.values

    rank = np.load(os.path.join('results',area+'_rank.npy'))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowrite = cv2.VideoWriter(area+'_pano.avi',fourcc,1,(1664,832))

    pano_array = []
    for idx in key_frame:
        panoid = info1[idx][0]
        pano_path = os.path.join(data_path, ('jpegs_' + city + '_2019'), panoid + '.jpg')
        pano = cv2.imread(pano_path)
        pano_array.append(pano)

    for i in range(len(pano_array)):
        videowrite.write(pano_array[i])
    videowrite.release()

    # save top1
    videowrite = cv2.VideoWriter(area+'_top1.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in key_frame:
        correspondence = rank[idx]
        corr = correspondence[0]
        tile_path = os.path.join('results', area, str(idx), str(corr)+'_0.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

    # save top2
    videowrite = cv2.VideoWriter(area+'_top2.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in key_frame:
        correspondence = rank[idx]
        corr = correspondence[1]
        tile_path = os.path.join('results', area, str(idx), str(corr)+'_1.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

    # save top3
    videowrite = cv2.VideoWriter(area+'_top3.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in key_frame:
        correspondence = rank[idx]
        corr = correspondence[2]
        tile_path = os.path.join('results', area, str(idx), str(corr)+'_2.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

    # save top4
    videowrite = cv2.VideoWriter(area+'_top4.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in key_frame:
        correspondence = rank[idx]
        corr = correspondence[3]
        tile_path = os.path.join('results', area, str(idx), str(corr)+'_3.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

    # save top5
    videowrite = cv2.VideoWriter(area+'_top5.avi',fourcc,1,(256,256))
    tile_array = []
    for idx in key_frame:
        correspondence = rank[idx]
        corr = correspondence[4]
        tile_path = os.path.join('results', area, str(idx), str(corr)+'_4.png')
        tile = cv2.imread(tile_path)
        tile_array.append(tile)

    for i in range(len(tile_array)):
        videowrite.write(tile_array[i])
    videowrite.release()

   
    






    




