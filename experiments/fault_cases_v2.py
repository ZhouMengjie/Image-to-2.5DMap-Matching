from doctest import OutputChecker
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.getcwd())
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import yaml
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from PIL import Image
from data import augmentation_simple

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
        dist = -2 * np.matmul(xyz, centroid)
        dist += xyz2
        dist +=  np.sum(centroid ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pt = point[centroids.astype(np.int32)]
    ft = feature[centroids.astype(np.int32)]
    return pt, ft

def load_and_process_data(model_name, location_name):
    file_path = os.path.join('ranks', location_name+'_'+model_name+'_rank.npy')
    data = np.load(file_path)
    success_indices = np.where(data[:, 0] == np.arange(data.shape[0]))[0]  
    success_rate = len(success_indices)/data.shape[0]*100
    return data, success_indices, success_rate

def save_tile(info1, info2, area, points, semantic_ids, colors, corr, out_path, file_name):
    panoid = info1[corr][0]
    center_x = info1[corr][1]
    center_y = info1[corr][2]
    heading = info1[corr][3]
    vertex_indices = np.load(os.path.join(data_path, (area+'U_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]
    feats = semantic_ids[vertex_indices]
    colors = []
    for j in range(len(feats)):
        class_id = feats[j][0]
        colors.append(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(colors)   
    pc = torch.tensor(coords, dtype=torch.float)
    center = [center_x, 0, center_y]
    result = {'heading': heading, 'center':center}
    result['cloud'] = torch.tensor(coords, dtype=torch.float)
    result['cloud_ft'] = torch.tensor(colors, dtype=torch.float)
    t = [augmentation_simple.RandomRotation(max_theta=0, axis=np.asarray([0,1,0])),
        augmentation_simple.RandomCenterCrop(radius=76, rnd=0)]
    transform = transforms.Compose(t)    
    result = transform(result)
    coords = result['cloud'].numpy()
    feats = result['cloud_ft'].numpy()
    coords, feats = farthest_point_sample(coords, feats, 1024)
    result['cloud'] = torch.tensor(coords, dtype=torch.float)
    result['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
    new_coords = result['cloud'].numpy()
    new_colors = result['cloud_ft'].numpy()

    # 3D to 2D projection
    center_x = torch.tensor(center_x)
    center_y = torch.tensor(center_y)
    image_w, image_h = 256, 256
    sensor_w, sensor_h = 152, 152
    pc = result['cloud']   # N x C
    cx = 0.5*sensor_w - center_x + 1
    cy = 0.5*sensor_h - center_y + 1
    image_x = torch.round((pc[:,0] + cx)*(image_w - 1) / (sensor_w))
    image_y = torch.round((pc[:,2] + cy)*(image_h - 1) / (sensor_h))
    image = torch.cat([
        image_x[:,None],
        image_y[:,None],
    ], dim=1)
    image = image.type(torch.int)

    # tile
    global_idx = info2[corr][1]
    city = info2[corr][0]
    tile_pathname = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    tile = cv2.imread(tile_pathname)
    point_list = image.numpy()
    for point in point_list:
        cv2.circle(tile,point,1,(255,0,0),1)
    cv2.imwrite(os.path.join(out_path,file_name),tile)

if __name__ == '__main__':
    location_name = 'wallstreet5k'
    model_name = '2d'
    baseline, baseline_success, baseline_recall = load_and_process_data(model_name, location_name)

    model_name = 'dgcnn2to3'
    ours, ours_success, ours_recall = load_and_process_data(model_name, location_name)

    all = np.arange(baseline.shape[0])
    cases = set(all) - set(ours_success) - set(baseline_success)
    # cases = set(ours_success) - set(baseline_success)
    idx = random.choice(list(cases))
    print('Ground Truth:', idx)
    print('Ours:', ours[idx][0])
    print('Ours-recall:', ours_recall)
    print('Baseline:', baseline[idx][0])
    print('Baseline-recall:', baseline_recall)

    # save examples
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = pd.read_csv(os.path.join(data_path, 'csv', (location_name + 'U_xy.csv')), sep=',', header=None)
    info1 = data.values
    label = pd.read_csv(os.path.join('datasets', 'csv', (location_name+ 'U_set.csv')), sep=',', header=None)
    info2 = label.values

    out_path = os.path.join('rank_examples','chp6',str(idx))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    panoid = info1[idx][0]
    city = info1[idx][4]
    img_path = os.path.join(data_path, ('jpegs_' + city + '_2019'), panoid + '.jpg')
    pano = Image.open(img_path)
    pano.save(os.path.join(out_path,'pano.png'))

    # load all points and semantic labels
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    classes = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = classes.values
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)
  
    save_tile(info1, info2, location_name, points, semantic_ids, color, idx, out_path, 'GT.png')
    save_tile(info1, info2, location_name, points, semantic_ids, color, ours[idx][0], out_path, 'ours.png')
    save_tile(info1, info2, location_name, points, semantic_ids, color, baseline[idx][0], out_path, 'baseline.png')
    









