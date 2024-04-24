import os
import random
import numpy as np

def load_and_process_data(pre_model_name, model_type, location_name):
    output_folder = os.path.join('results', pre_model_name)
    file_path = os.path.join(output_folder, f"{location_name}_{model_type}_rank.npy") 
    data = np.load(file_path)
    success_indices = np.where(data[:, 0] == np.arange(data.shape[0]))[0]  
    return data, success_indices

if __name__ == '__main__':
    location_name = 'wallstreet5kU'
    pre_model_name = 'resnetsafa_dgcnn_asam_2to3_up_16'
    model_type = 'baseline'
    baseline, baseline_success = load_and_process_data(pre_model_name, model_type, location_name)

    pre_model_name = 'resnetsafa_dgcnn_asam_2to3_up'
    model_type = 'transmixer'
    transmixer, transmixer_success = load_and_process_data(pre_model_name, model_type, location_name)

    # model_type = 'seqnet'
    # seqnet, seqnet_success = load_and_process_data(pre_model_name, model_type, location_name)

    all = np.arange(baseline.shape[0])
    success = set(all) - set(baseline_success) - set(transmixer_success)
    # success = set(transmixer_success) - set(baseline_success)

    ndx = random.choice(list(success))
    print('Ground Truth:', ndx)
    print('Transmixer:', transmixer[ndx][0])
    # print('SeqNet:', seqnet[ndx][0])
    print('Baseline:', baseline[ndx][0])
    









