import os
import random
import numpy as np

def load_and_process_data(file_path):
    data = np.load(file_path)
    success_indices = np.where(data[:, 0] == np.arange(data.shape[0]))[0]  
    return data, success_indices

if __name__ == '__main__':
    location_name = 'unionsquare5kU'
    pre_model_name = 'resnetsafa_dgcnn_asam_2to3_up'
    model_type = 'seqnet'
    output_folder = os.path.join('results', pre_model_name)
    
    file_path = os.path.join(output_folder, f"{location_name}_{model_type}_rank.npy") 
    base, base_success = load_and_process_data(file_path)

    
    file_path = os.path.join(output_folder, f"{location_name}_{model_type}_sfrank.npy") 
    shuffle, shuffle_success = load_and_process_data(file_path)

    intersect = set(base_success) & set(shuffle_success)

    print('Success cases:', len(base_success))
    print('FP cases:', len(intersect))
    
    intersect = list(intersect)
    intersect.sort()
    # ndx = random.choice(intersect)
    ndx = 0
    print('Ground Truth:', ndx)
    print('Base:', base[ndx][0])
    print('Shuffle:', shuffle[ndx][0])
    









