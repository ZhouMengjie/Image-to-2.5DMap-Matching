import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.getcwd())
import yaml
import pandas as pd
import numpy as np
import torch
import open3d as o3d
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from experiments.visualizer import visualize_pcd  
from openstreetmap import cropping
from data import augmentation_pc
import torchvision.transforms as transforms

if __name__ == "__main__":
    # load pcd of the whole map
    data_path = 'datasets'
    # load csv file including pano_id, center (x,y), and heading angle
    area = 'unionsquare5kU'
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values

    # deploy augmentation on local pcd
    idx = 100
    # idx = random.randint(0, info.shape[0]-1)
    panoid = info[idx][0]
    center_x = info[idx][1]
    center_y = info[idx][2]
    heading = info[idx][3] # in degree
    city = info[idx][4]

    # load point cloud
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    # load semantic id 
    data = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = data.values
    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    # original 228*228 area
    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]
    feats = semantic_ids[vertex_indices]
    colors = []
    for i in range(len(feats)):
        class_id = feats[i][0]
        colors.append(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(colors)
    visualize_pcd(coords, colors, 'npy')

    # to tensor
    pc = torch.tensor(coords, dtype=torch.float)

    # JitterPoints
    # transform = augmentation.JitterPoints(sigma=10, clip=10)
    # new_coords = transform(pc).numpy()
    # visualize_pcd(new_coords, colors, 'npy')

    # RemoveRandomPoints
    # transform = augmentation.RemoveRandomPoints(r=(0.0, 1))
    # new_coords, saved_mask = transform(pc)
    # new_coords = new_coords.numpy()
    # new_colors = colors[saved_mask][:]
    # visualize_pcd(new_coords, new_colors, 'npy')

    # RemoveRandomBlock
    # transform = augmentation.RemoveRandomBlock(p=1)
    # new_coords, saved_mask = transform(pc)
    # new_coords = new_coords.numpy()
    # new_colors = colors[saved_mask][:]
    # visualize_pcd(new_coords, new_colors, 'npy')

    # RandomTranslation
    # transform = augmentation.RandomTranslation(max_delta=5)
    # new_coords = transform(pc).numpy()
    # visualize_pcd(new_coords, colors, 'npy')

    # RandomFlip
    # transform = augmentation.RandomFlip([0.5, 0, 0.5])
    # new_coords = transform(pc).numpy()
    # visualize_pcd(new_coords, colors, 'npy')

    # RandomRotation
    # center = [center_x, 0, center_y]
    # # print(points_o3d.get_rotation_matrix_from_xyz((0, heading, 0)))
    # transform = augmentation.RandomRotation(max_theta=heading, max_theta2=0, center=center, axis=np.array([0, 1, 0]))
    # new_coords = transform(pc)
    # visualize_pcd(new_coords, colors, 'npy')


    # RandomScale
    # center = [center_x, center_y]
    # transform = augmentation.RandomScale(radius=76, rnd=5, center=center)
    # new_coords, saved_mask = transform(pc)
    # new_coords = new_coords.numpy()
    # new_colors = colors[saved_mask][:]
    # visualize_pcd(new_coords, new_colors, 'npy')

    # rotation + crop
    center = [center_x, 0, center_y]
    result = {'heading': heading, 'center':center}
    result['cloud'] = torch.tensor(coords, dtype=torch.float)
    result['cloud_ft'] = torch.tensor(colors, dtype=torch.float)
    t = [augmentation_pc.RandomRotation(max_theta=0, axis=np.asarray([0,1,0])),
        augmentation_pc.RandomCenterCrop(radius=76, rnd=0)]
    transform = transforms.Compose(t)
    
    result = transform(result)
    new_coords = result['cloud'].numpy()
    new_colors = result['cloud_ft'].numpy()
    visualize_pcd(new_coords, new_colors, 'npy')

    # rotate + translation + crop
    # center = [center_x, 0, center_y]
    # t = [augmentation.RandomRotation(max_theta=heading, max_theta2=0, center=center, axis=np.array([0, 1, 0])),
    #     augmentation.RandomTranslation(max_delta=10),
    #     augmentation.RandomCenterCrop(radius=76, rnd=0, center=center)]
    # transform = transforms.Compose(t)
    
    # new_coords, saved_mask = transform(pc)
    # new_coords = new_coords.numpy()
    # new_colors = colors[saved_mask][:]
    # visualize_pcd(new_coords, new_colors, 'npy')

    # rotation + crop + scale
    # center = [center_x, 0, center_y]
    # t = [augmentation.RandomRotation(max_theta=heading, max_theta2=0, center=center, axis=np.array([0, 1, 0])),
    #     augmentation.RandomCenterCrop(radius=64, rnd=0, center=center),
    #     augmentation.RandomCenterScale(src=64, dst=76, rnd=0, center=center)]
    # transform = transforms.Compose(t)
    # new_coords, saved_mask = transform(pc)
    # new_coords = new_coords.numpy()
    # new_colors = colors[saved_mask][:]
    # visualize_pcd(new_coords, new_colors, 'npy')    

    # check overlap
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(new_coords)
    # pcd.colors = o3d.utility.Vector3dVector(new_colors)

    # pcdo = o3d.geometry.PointCloud()
    # pcdo.points = o3d.utility.Vector3dVector(o_coords)
    # pcdo.colors = o3d.utility.Vector3dVector(o_colors)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="pcl")
    # vis.get_render_option().point_size = 1
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([1, 1, 1])
    # vis.add_geometry(pcd)
    # vis.add_geometry(pcdo)
    # vis.run()
    # vis.destroy_window()

    print('debug')

 






