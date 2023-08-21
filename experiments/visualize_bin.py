import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
from experiments.visualizer import visualize_pcd  

if __name__ == "__main__":
    pbin = np.fromfile(os.path.join('datasets', ('1400505893170765.bin'))).reshape(-1,3)
    # npy to pcd
    pcd=o3d.geometry.PointCloud()
    pcd.points= o3d.utility.Vector3dVector(pbin)
    visualize_pcd(pcd, None, 'pcd')