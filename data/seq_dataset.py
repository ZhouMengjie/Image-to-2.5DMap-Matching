import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, dataset_path: str, query_filename: str, model_name: str):
        self.dataset_path = dataset_path
        self.seq_filepath = os.path.join(dataset_path, 'csv', query_filename+'_sq.csv')
        self.sequences = (pd.read_csv(self.seq_filepath, sep=',', header=None)).values
        self.indexes = np.arange(0, len(self.sequences))

        self.pano_filepath = os.path.join('datasets', 'features', 'pano', query_filename+'_'+model_name+'.npy')
        self.pano_descriptors = np.load(self.pano_filepath)
        self.map_filepath = os.path.join('datasets', 'features', 'map', query_filename+'_'+model_name+'.npy')
        self.map_descriptors = np.load(self.map_filepath)
        print('{} queries in the dataset'.format(len(self)))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, ndx):
        # Load single precomputed descriptors for each sequence
        pano_seq = []
        map_seq = []
        seq_indices = self.sequences[ndx]
        for idx in seq_indices:
            pano_seq.append(torch.tensor(self.pano_descriptors[idx], dtype=torch.float))
            map_seq.append(torch.tensor(self.map_descriptors[idx], dtype=torch.float))
        pano_seq = torch.stack(pano_seq, dim=0)
        map_seq = torch.stack(map_seq, dim=0)
        return {'pano':pano_seq, 'map':map_seq, 'label':torch.tensor(ndx)}

if __name__ == '__main__':
    dataset_path = 'datasets'
    query_filename = 'unionsquare5kU'
    model_name = 'resnetsafa_asam_simple'
    dataset = {}
    dataset['train'] = SeqDataset(dataset_path, query_filename, model_name)
    batch_sampler = None
    batch_size = 16
    nw = 0
    device = 'cuda'
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset['train'], batch_sampler=batch_sampler, batch_size=batch_size,
                            num_workers=nw, pin_memory=True, shuffle=True, drop_last=True)
    for batch in dataloaders['train']:
        batch = {e: batch[e].to(device) for e in batch}
        print(batch['pano'].shape)
        print(batch['map'].shape)
        print(batch['label'].shape)

        



   

    
   

    
    
    
        

