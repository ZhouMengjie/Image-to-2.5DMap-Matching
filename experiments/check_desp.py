import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == '__main__':
        # query_filename = 'trainstreetlearnU_cmu5kU'
        # model_name = 'resnetsafa_dgcnn_asam_2to3_up'
        # pano_filepath = os.path.join('datasets', 'features', 'pano', query_filename+'_'+model_name+'.npy')
        # pano_descriptors = np.load(pano_filepath)
        # map_filepath = os.path.join('datasets', 'features', 'map', query_filename+'_'+model_name+'.npy')
        # map_descriptors = np.load(map_filepath)



        # Create a tensor
        cloud_embedding = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],[7.0, 8.0, 9.0]])

        # Normalize along dimension 1 (rows)
        normalized_tensor = F.normalize(cloud_embedding, p=2, dim=1)
        print(normalized_tensor)