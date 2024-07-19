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
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
from PIL import Image
from data import augmentation_simple

def load_and_process_data(model_name, location_name):
    file_path = os.path.join('ranks', location_name+'_'+model_name+'_rank.npy')
    data = np.load(file_path)
    success_indices = np.where(data[:, 0] == np.arange(data.shape[0]))[0]  
    success_rate = len(success_indices)/data.shape[0]*100
    return data, success_indices, success_rate

if __name__ == '__main__':
    location_name = 'wallstreet5k'
    model_name = '2d'
    baseline, baseline_success, baseline_recall = load_and_process_data(model_name, location_name)

    model_name = '2dsafapolar'
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

    out_path = os.path.join('rank_examples','chp5',str(idx))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    panoid = info1[idx][0]
    city = info1[idx][4]
    img_path = os.path.join(data_path, ('jpegs_' + city + '_2019'), panoid + '.jpg')
    pano = Image.open(img_path)
    pano.save(os.path.join(out_path,'pano.png'))

    global_idx = info2[idx][1]
    city = info2[idx][0]
    tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_path)
    tile.save(os.path.join(out_path,'GT.png'))

    global_idx = info2[ours[idx][0]][1]
    city = info2[ours[idx][0]][0]
    tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_path)
    tile.save(os.path.join(out_path,'ours.png'))

    global_idx = info2[baseline[idx][0]][1]
    city = info2[baseline[idx][0]][0]
    tile_path = os.path.join('datasets', 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
    tile = Image.open(tile_path)
    tile.save(os.path.join(out_path,'baseline.png'))

    









