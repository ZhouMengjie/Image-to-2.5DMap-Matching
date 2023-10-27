import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
        query_filename = 'hudsonriver5kU'
        model_name = 'resnetsafa_asam_simple'
        pano_filepath = os.path.join('datasets', 'features', 'pano2', query_filename+'_'+model_name+'.npy')
        pano_descriptors = np.load(pano_filepath)
        map_filepath = os.path.join('datasets', 'features', 'map', query_filename+'_'+model_name+'.npy')
        map_descriptors = np.load(map_filepath)

