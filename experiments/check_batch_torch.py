import os
import sys
import torch
sys.path.append(os.getcwd())
from PIL import Image
import pandas as pd
import yaml
import random
import torchvision.transforms as transforms
import numpy as np
from experiments.visualizer import visualize_pcd  

def tensor2img(x):
    t = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                            transforms.ToPILImage()])
    return t(x)


if __name__ == "__main__":
    data_folder = 'train_batch'
    batch_id = 1
    # random pick a data from one batch
    # seed = random.randint(0, 31)
    seed = 24
    print(seed)

    area = 'trainstreetlearnU'
    data = pd.read_csv(os.path.join('datasets', 'csv', ( area+ '_set.csv')), sep=',', header=None)
    info = data.values
    
    #-------------gpu--------------#
    device = '3'
    coords = np.load(os.path.join(data_folder, 'coords_'+str(batch_id)+'_'+device+'.npy'))
    features = np.load(os.path.join(data_folder, 'features_'+str(batch_id)+'_'+device+'.npy'))
    images = np.load(os.path.join(data_folder, 'images_'+str(batch_id)+'_'+device+'.npy'))
    labels = np.load(os.path.join(data_folder, 'labels_'+str(batch_id)+'_'+device+'.npy'))
    
    with open('color_map.yaml','r') as f:
        colors = yaml.load(f, Loader=yaml.FullLoader)
    
    coord = []
    color = []
    pcd = coords[seed].T
    for i in range(len(pcd)):
        coord.append(pcd[i][:3])
        class_id = np.argmax(pcd[i][3:])
        color.append(np.divide(colors['color_map'][class_id], 255))
    coord = np.asarray(coord)
    color = np.asarray(color)
    visualize_pcd(coord, color, 'npy')
    
    # show pano
    img = images[seed]
    img_tensor = torch.tensor(img)
    x = tensor2img(img_tensor)
    x.show()
    x.save(os.path.join(data_folder, str(batch_id)+'_'+str(seed)+'_'+device+'.jpeg'))

    # show tile
    idx = labels[seed]
    global_idx = info[idx][1] # for train only
    city = info[idx][0]
    tile_path = os.path.join('datasets', city+'_z18', str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_path)
    tile.show()
    tile.save(os.path.join(data_folder, str(idx)+'.png'))   





