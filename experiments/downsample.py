import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import open3d as o3d
from experiments.visualizer import visualize_pcd


if __name__ == "__main__":
    city = 'pittsburgh'
    data_path = os.path.join(os.getcwd(), 'datasets', city)
    print('Displaying the completed point cloud ...')
    # pcd = o3d.io.read_point_cloud(os.path.join(data_path, (city + '.pcd')))

    # load pcd of the whole map
    data_path = 'datasets'
    points = np.load(os.path.join(data_path, 'manhattan', 'manhattan.npy'))
    # load csv file including pano_id, center (x,y), and heading angle
    area = 'wallstreet5k'
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    # deploy augmentation on local pcd
    idx = 0
    panoid = info[idx][0]
    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.paint_uniform_color([1, 0.706, 0])
    pcdd = pcd.voxel_down_sample(voxel_size=5)
    visualize_pcd(pcdd, None, 'pcd')