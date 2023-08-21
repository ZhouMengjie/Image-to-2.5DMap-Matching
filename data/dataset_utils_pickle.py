import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.augmentation_simple import TrainTransform, ValTransform, TrainRGBTransform, ValRGBTransform, TrainTileTransform, ValTileTransform
from config.utils import MinkLocParams
from data.streetlearn_no_mc import StreetLearnDataset

def make_datasets(params: MinkLocParams):
    # Create training and validation datasets
    datasets = {}

    if params.use_cloud:
        train_transform = TrainTransform(params.aug_mode)
        val_transform = ValTransform()
    else:
        train_transform = None
        val_transform = None

    if params.use_rgb:
        image_train_transform = TrainRGBTransform(params.aug_mode)
        image_val_transform = ValRGBTransform()
    else:
        image_train_transform = None
        image_val_transform = None

    if params.use_tile:
        tile_train_transform = TrainTileTransform(params.aug_mode, params.model_params.tile_size)
        tile_val_transform = ValTileTransform(params.model_params.tile_size)
    else:
        tile_train_transform = None
        tile_val_transform = None


    
    datasets['train'] = StreetLearnDataset(params.dataset_folder, params.train_file,
                                           transform=train_transform, 
                                           image_size=params.model_params.img_size, image_transform=image_train_transform,
                                           tile_size=params.model_params.tile_size, tile_transform=tile_train_transform,
                                           use_cloud=params.use_cloud, use_rgb=params.use_rgb, use_tile=params.use_tile,
                                           use_feat=params.use_feat, use_polar=params.use_polar, 
                                           normalize=params.pc_normalize, npoints=params.npoints)
                                      
    if params.use_val:
        datasets['val'] = StreetLearnDataset(params.dataset_folder, params.val_file,
                                            transform=val_transform,
                                            image_size=params.model_params.img_size, image_transform=image_val_transform,
                                            tile_size=params.model_params.tile_size, tile_transform=tile_val_transform,
                                            use_cloud=params.use_cloud, use_rgb=params.use_rgb, use_tile=params.use_tile,
                                            use_feat=params.use_feat, use_polar=params.use_polar, 
                                            normalize=params.pc_normalize, npoints=params.npoints)
    return datasets


def make_collate_fn_torch(dataset: StreetLearnDataset):
    def collate_fn(data_list):

        result = {}
        labels = [d[0]['ndx'] for d in data_list] 
        result['labels'] = torch.tensor(labels)
        center = [d[0]['center'] for d in data_list]
        result['center'] = torch.tensor(center)
        if dataset.transform.aug_mode == 1:
            result['center'] = torch.tensor(center).repeat(2,1)

        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        result['positives_mask'] = torch.tensor(positives_mask)
        result['negatives_mask'] = torch.tensor(negatives_mask)

        if dataset.use_rgb:
            images_batch = torch.stack([d[0]["image"] for d in data_list])
            if dataset.image_transform.aug_mode == 1:
                images_aug_batch = torch.stack([d[1]["image"] for d in data_list])
                images_batch = torch.cat((images_batch, images_aug_batch))
            result['images'] = images_batch

        if dataset.use_cloud:
            coordinates_batch = torch.stack([d[0]["cloud"].T for d in data_list])
            xyz_batch = torch.stack([d[0]["xyz"].T for d in data_list])
            if dataset.transform.aug_mode == 1:
                coordinates_aug_batch = torch.stack([d[1]["cloud"].T for d in data_list])
                xyz_aug_batch = torch.stack([d[1]["xyz"].T for d in data_list])
                coordinates_batch = torch.cat((coordinates_batch, coordinates_aug_batch))
                xyz_batch = torch.cat((xyz_batch, xyz_aug_batch))
            result['coords'] = coordinates_batch
            result['xyz'] = xyz_batch

        if dataset.use_tile:
            tiles_batch = torch.stack([d[0]["tile"] for d in data_list])
            if dataset.tile_transform.aug_mode == 1:
                tiles_aug_batch = torch.stack([d[1]["tile"] for d in data_list])
                tiles_batch = torch.cat((tiles_batch, tiles_aug_batch))
            result['tiles'] = tiles_batch

        return result
    return collate_fn


def make_dataloaders(params: MinkLocParams):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params)
    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers
    # nw = 0 # for cpu debug
    if params.log:
        print('Using {} dataloader workers every process'.format(nw))

    dataloaders = {}
    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(datasets['train'], num_replicas=params.world_size, rank=params.rank, shuffle=True)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, params.batch_size, drop_last=True)
        train_collate_fn = make_collate_fn_torch(datasets['train'])
        dataloaders['train'] = DataLoader(datasets['train'], batch_sampler=train_batch_sampler, collate_fn=train_collate_fn,
                                        num_workers=nw, pin_memory=True)
    else:
        train_batch_sampler = None
        train_collate_fn = make_collate_fn_torch(datasets['train'])
        dataloaders['train'] = DataLoader(datasets['train'], batch_sampler=train_batch_sampler, batch_size=params.batch_size, collate_fn=train_collate_fn,
                                        num_workers=nw, pin_memory=True, shuffle=True, drop_last=True)
    if 'val' in datasets:
        if params.val_distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(datasets['val'], num_replicas=params.world_size, rank=params.rank, shuffle=False)
            val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, params.val_batch_size, drop_last=False)
            val_collate_fn = make_collate_fn_torch(datasets['val'])
            dataloaders['val'] = DataLoader(datasets['val'], sampler=val_batch_sampler, collate_fn=val_collate_fn,
                                          num_workers=nw, pin_memory=True)           
        else:
            val_batch_sampler = None
            val_collate_fn = make_collate_fn_torch(datasets['val'])
            dataloaders['val'] = DataLoader(datasets['val'], sampler=val_batch_sampler, batch_size=params.val_batch_size, collate_fn=val_collate_fn,
                                num_workers=nw, pin_memory=True, shuffle=False, drop_last=False)
            
    if params.distributed:
        return dataloaders, train_sampler
    else:
        return dataloaders, None


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e



