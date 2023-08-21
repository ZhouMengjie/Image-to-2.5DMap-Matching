import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import torch
import random
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from data import augmentation_img
from data import Equirec2Perspec as E2P


if __name__ == "__main__":
    # check all images
    # img_files= os.listdir(os.path.join('datasets', 'jpegs_pittsburgh_2019'))
    # for i, img_file in enumerate(img_files):
    #     print(i)
    #     if img_file is not 'links.txt' and img_file is not 'nodes.txt':
    #         pano = cv2.imread(os.path.join('data', 'jpegs_pittsburgh_2019', img_file))
    #         zoom = int(np.ceil(pano.shape[0] / 512))

    #         if  pano.shape[0] > 512*np.power(2,zoom-1):
    #             print(img_file)


    # load csv file including pano_id, center (x,y), and heading angle
    data_path = 'datasets'
    area = 'wallstreet5k'
    data = pd.read_csv(os.path.join(data_path, 'csv', (area + '_xy.csv')), sep=',', header=None)
    info = data.values

    # deploy augmentation on local pano or snapshot
    idx = 0
    # idx = random.randint(0, info.shape[0]-1)
    panoid = info[idx][0]
    center_x = info[idx][1]
    center_y = info[idx][2]
    heading = info[idx][3] # in degree
    city = info[idx][4]

    image_ext = '.jpg'
    image_file_path = os.path.join(data_path, 'jpegs_'+city+'_2019', panoid + image_ext)


    # PIL
    # pano = Image.open(image_file_path)
    # pano.show()
    # pano.save(os.path.join('results', panoid+'_0.jpg'))

    # colorjitter
    # t = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # img_color = t(pano)
    # img_color.show()
    # img_color.save(os.path.join('results', panoid+'_color.jpg'))

    # normalize
    # t = transforms.Compose([transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # img_t = t(pano)

    # randomerasing
    # t = transforms.Compose([transforms.ToTensor(), 
    #                         transforms.RandomErasing(scale=(0.1, 1.0)),
    #                         transforms.ToPILImage()])
    # img_t = t(pano)
    # img_t.show()
    # img_t.save(os.path.join('results', panoid+'_erase'+'.jpg'))

    # normalize
    # t = transforms.Compose([transforms.ToTensor(), 
    #                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #                         transforms.ToPILImage()])
    # img_t = t(pano)
    # img_t.show()
    # img_t.save(os.path.join('results', panoid+'_normalize'+'.jpg'))

    # noise
    # t = transforms.Compose([transforms.ToTensor(), 
    #                         augmentation_v2.AddGaussianNoise(mean=0.0, std=0.01), 
    #                         transforms.ToPILImage()])
    # img_t = t(pano)
    # img_t.show()
    # img_t.save(os.path.join('results', panoid+'_noise'+'.jpg'))

    # augmentation_v2.tensor2img(img_t).show()

    # cv2
    # pano 
    pano_cv2 = cv2.imread(image_file_path)
    cv2.imshow(panoid, pano_cv2)
    cv2.waitKey(0)

    # pano rotate
    yaw  = -20 # A slighly random rotation
    pitch = 0
    roll = 0
    print([yaw, pitch, roll])
    pano_t = augmentation_img.rotate_panorama(pano_cv2, roll, pitch, yaw)
    cv2.imshow(panoid, pano_cv2)
    cv2.waitKey(0)
    pano_t = cv2.cvtColor(pano_t, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pano_t)
    img.show()
    img.save(os.path.join('results', panoid+'_'+str(yaw)+'.jpg'))

    # snap
    snaps = []
    equ = E2P.Equirectangular(pano_cv2)
    views = [0,-90,90,180] 
    size = 224        
    H, W = size if hasattr(size,'__iter__') else (size,size)

    fov_shift = 0
    pitch_shift = 0
    tetha_shift = -20
   
    snaps = [equ.GetPerspective(90+fov_shift, t+tetha_shift, pitch_shift, H, W) for t in views]
    # [F, L, R, B]
    for i, img in enumerate(snaps): 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB and change range to [0-1]
        img = Image.fromarray(img)
        img.show()
        img.save(os.path.join('results', panoid+'_d'+str(i)+'.jpg'))
        
    print('debug')
