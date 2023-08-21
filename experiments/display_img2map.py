import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.getcwd())
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import yaml
from experiments.visualizer import visualize_pcd  
from data import augmentation_pc
from PIL import Image
from data import Equirec2Perspec as E2P
import cv2

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'datasets')
    area = 'unionsquare5kU'
    if not os.path.isdir(os.path.join('results', area)):
        os.makedirs(os.path.join('results', area))

    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values
    idx = random.randint(0, info.shape[0]-1)
    idx = 1000
    # print(idx)
    panoid = info[idx][0]
    center_x = info[idx][1]
    center_y = info[idx][2]
    heading = info[idx][3]
    city = info[idx][4]
    # print(city)

    # load point cloud
    points = np.load(os.path.join(data_path, city, city+'U.npy'))
    # load semantic id 
    data = pd.read_csv(os.path.join(data_path, city, city+'U.csv'), sep=',', header=None)
    semantic_ids = data.values
    # load color map
    with open('color_map.yaml','r') as f:
        color = yaml.load(f, Loader=yaml.FullLoader)

    vertex_indices = np.load(os.path.join(data_path, (area+'_idx'), (panoid + '.npy')))
    coords = points[vertex_indices][:]
    feats = semantic_ids[vertex_indices]
    colors = []
    for i in range(len(feats)):
        class_id = feats[i][0]
        colors.append(np.divide(color['color_map'][class_id], 255))
    colors = np.asarray(colors)
    
    pc = torch.tensor(coords, dtype=torch.float)
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
    # visualize_pcd(new_coords, new_colors, 'npy')

    # load image and map tile
    img_path = os.path.join(data_path, ('jpegs_' + city + '_2019'), panoid + '.jpg')
    pano = Image.open(img_path)
    pano.show()
    pano.save(os.path.join('results',panoid+'_pano.png'))
    # pano = cv2.imread(img_path)
    # snaps = []
    # equ = E2P.Equirectangular(pano)
    # views = [0,-90,90,180] 
    # size = 224        
    # H, W = size if hasattr(size,'__iter__') else (size,size)   
    # snaps = [equ.GetPerspective(100, t, 0, H, W) for t in views]
    # cv2.imwrite('results/snap0.png',snaps[0])
    # cv2.imwrite('results/snap1.png',snaps[1])
    # cv2.imwrite('results/snap2.png',snaps[2])
    # cv2.imwrite('results/snap3.png',snaps[3])

    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_set.csv')), sep=',', header=None)
    info = data.values
    global_idx = info[idx][1] # for train only
    city = info[idx][0]
    tile_pathname = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_pathname)
    tile.show()
    tile.save(os.path.join('results',panoid+'_map.png')) 

    # load and display panorama and 2D map-tile
    # img_path = os.path.join(data_path, ('jpegs_' + city + '_2019'), panoid + '.jpg')
    # pano = imgplt.imread(img_path)
    # snaps = []
    # equ = E2P.Equirectangular(pano)
    # views = [0,-90,90,180] 
    # size = 224        
    # H, W = size if hasattr(size,'__iter__') else (size,size)   
    # snaps = [equ.GetPerspective(90, t, 0, H, W) for t in views]

    # tile
    # data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_set.csv')), sep=',', header=None)
    # info = data.values
    # if area == 'trainstreetlearn':
    #     global_idx = int(info[idx+1][1])-1 # for train only
    # else:
    #     global_idx = info[idx+1][1] # for train only
    # tile_path = os.path.join(data_path, city+'_z18', str(global_idx).zfill(5) + '.png')
    # tile = imgplt.imread(tile_path)

    # grid = plt.GridSpec(2, 4)
    # plt.subplot(grid[0,0:3])
    # plt.title(panoid)
    # plt.imshow(pano)
    # plt.axis('off')

    # plt.subplot(grid[0,3:4])
    # plt.title('2D map')
    # plt.imshow(tile)
    # plt.axis('off')

    # # front
    # plt.subplot(grid[1,0])  
    # plt.title('front') 
    # plt.imshow(snaps[0])
    # plt.axis('off')

    # # left
    # plt.subplot(grid[1,1])  
    # plt.title('left')
    # plt.imshow(snaps[1])
    # plt.axis('off')

    # # right
    # plt.subplot(grid[1,2])  
    # plt.title('right')
    # plt.imshow(snaps[2])
    # plt.axis('off')

    # # back
    # plt.subplot(grid[1,3])  
    # plt.title('back')
    # plt.imshow(snaps[3])
    # plt.axis('off')
    # plt.savefig(os.path.join("results", area, (panoid + ".jpeg")), bbox_inches='tight')
    # plt.waitforbuttonpress(0)



