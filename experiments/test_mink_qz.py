import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import torch
import MinkowskiEngine as ME


if __name__ == "__main__":
    # load pcd of the whole map
    data_path = 'datasets'
    points = np.load(os.path.join(data_path, 'manhattan', 'manhattan.npy'))
    # points_o3d = o3d.io.read_point_cloud(os.path.join(data_path, 'manhattan', 'manhattan50.pcd'))

    # load csv file including pano_id, center (x,y), and heading angle
    area = 'wallstreet5k'
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values

    # load local pcd
    idx = 0
    # idx = random.randint(0, info.shape[0]-1)
    panoid = info[idx][0]
    center_x = info[idx][1]
    center_y = info[idx][2]
    heading = info[idx][3] # in degree
    city = info[idx][4]
    
    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]
    qz_coords = ME.utils.sparse_quantize(coords, quantization_size=0.1)
    qz_coords = qz_coords.numpy()
    np.save(os.path.join(data_path, 'qz01.npy'), qz_coords)


