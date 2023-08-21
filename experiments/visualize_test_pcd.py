import os
import sys
sys.path.append(os.getcwd())
import yaml
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from functools import reduce
from openstreetmap.transverse_mercator import TransverseMercator
from visualizer import visualize_pcd

if __name__ == "__main__":
    city = 'manhattan'
    data_path = os.path.join(os.getcwd(), 'datasets', city)
    print('Displaying the completed point cloud ...')
    pcd = o3d.io.read_point_cloud(os.path.join(data_path, (city + '.pcd')))
    
    # crop pcd
    data_path = 'datasets'
    area = 'wallstreet5kU'
    queries = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None).values

    min_x = min(queries[:,1]) - 76
    min_y = min(queries[:,2]) - 76
    max_x = max(queries[:,1]) + 76
    max_y = max(queries[:,2]) + 76


    # area = 'wallstreet5k'
    # queries = pd.read_csv(os.path.join(data_path, 'csv', (area + '.csv')), sep=',', header=None).values
    # min_x = min(queries[:,1])
    # min_y = min(queries[:,2])
    # max_x = max(queries[:,1])
    # max_y = max(queries[:,2])
    # print((min_x,min_y))
    # print((max_x,max_y))

    vertex_x = np.asarray(pcd.points)[:,0]
    vertex_y = np.asarray(pcd.points)[:,2]

    indices_x1 = np.where(vertex_x > min_x)
    indices_x2 = np.where(vertex_x < max_x)
    indices_y1 = np.where(vertex_y > min_y)
    indices_y2 = np.where(vertex_y < max_y)
    vertex_indices = reduce(np.intersect1d,[indices_x1,indices_x2,indices_y1,indices_y2])

    points = np.asarray(pcd.points)[vertex_indices][:]
    colors = np.asarray(pcd.colors)[vertex_indices][:]
    normals = np.asarray(pcd.normals)[vertex_indices][:]
    pcd_test = o3d.geometry.PointCloud()
    pcd_test.points = o3d.utility.Vector3dVector(points)
    pcd_test.colors = o3d.utility.Vector3dVector(colors)
    pcd_test.normals = o3d.utility.Vector3dVector(normals)
    visualize_pcd(pcd_test, None, type='pcd')

    




