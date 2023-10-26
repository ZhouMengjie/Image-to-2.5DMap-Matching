import os
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
import sys
sys.path.append(os.getcwd())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from config.utils import get_datetime
from data.seq_dataset_v2 import SeqDataset_v2, make_collate_fn
from training_seq_v2.distributed_utils import init_distributed_mode
from training_seq_v2.trainer import do_train
from data.augmentation_simple import TrainTransform, ValTransform, TrainRGBTransform, ValRGBTransform, TrainTileTransform, ValTileTransform


def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('Seed: {}'.format(random_seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--dataset_folder', type=str, default='datasets', required=False, help='Dataset folder')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=24, required=False, help='Testing batch size')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--epoch', type=int, default=1, required=False, help='Initial training epoch')    
    parser.add_argument('--train_file', type=str, required=False, help='Train file')
    parser.add_argument('--val_file', type=str, required=False, help='Val file')
    parser.add_argument('--distributed', dest='distributed', action='store_true')
    parser.add_argument('--val_distributed', dest='val_distributed', action='store_true')
    parser.add_argument('--syncBN', dest='syncBN', action='store_true')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
    parser.add_argument('--share', dest='share', action='store_true')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Seed')
    parser.add_argument('--port', type=str, default='12363', required=False, help='Port')  
    parser.add_argument('--feat_dim',type=int, default=4096, required=False, help='Feature dimension')
    parser.add_argument('--margin', type=float, default=0.07, required=False, help='Loss margin')
    parser.add_argument('--optimizer', type=str, required=False, help='Optimizer')
    parser.add_argument('--wd', type=float, default=0.03, required=False, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=60, required=False, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, required=False, help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, required=False, help='LR Scheduler')
    parser.add_argument('--num_layers', type=int, default=1, required=False, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, required=False, help='Number of transformer heads')
    parser.add_argument('--seq_len', type=int, default=5, required=False, help='Sequence length')
    parser.add_argument('--model_type', type=str, required=False, help='Model type')

    # arguments for seqdataset_v2
    parser.add_argument('--use_cloud', dest='use_cloud', action='store_true') # always false
    parser.add_argument('--use_polar', dest='use_polar', action='store_true')
    parser.add_argument('--freeze', dest='freeze', action='store_true')
    parser.add_argument('--aug_mode', type=int, default=0, required=False, help='Whether use data augmentation')
    parser.add_argument('--tile_size', type=int, default=224, required=False, help='Size of tile')
    parser.add_argument('--image_size', type=int, default=224, required=False, help='Size of pano')
    parser.add_argument('--map_type', type=str, required=False, help='Map type')
    parser.add_argument('--pretrained', type=str, required=False, help='Pretrained encoder weights')
    parser.add_argument('--encoder_dim', type=int, default=512, required=False, help='Encoder output dimension')

    parser.set_defaults(train_file='trainstreetlearnU_cmu5kU')
    parser.set_defaults(val_file='hudsonriver5kU')
    parser.set_defaults(optimizer='SAM')
    parser.set_defaults(scheduler='CosineAnnealingLR')
    parser.set_defaults(model_type='transmixer')
    parser.set_defaults(map_type='multi')

    params = parser.parse_args()
    seed_all(params.seed)
    
    savedStdout = sys.stdout
    s = get_datetime()
    if not os.path.exists('arun_log'):
        os.mkdir('arun_log')
    print_log = open(os.path.join('arun_log',s+'.txt'),'w')
    sys.stdout = print_log

    if params.distributed:
        init_distributed_mode(params)
    else:
        params.log = True
        if torch.cuda.is_available():
            params.device = torch.device('cuda')
        else:
            params.device = torch.device('cpu')

    if params.log:
        print('Parameters:')
        param_dict = vars(params)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))
        print('')

    # make dataloaders
    if params.use_cloud:
        cloud_train_transform = TrainTransform(params.aug_mode)
        cloud_val_transform = ValTransform()
    else:
        cloud_train_transform = None
        cloud_val_transform = None

    image_train_transform = TrainRGBTransform(params.aug_mode)
    image_val_transform = ValRGBTransform()
    tile_train_transform = TrainTileTransform(params.aug_mode)
    tile_val_transform = ValTileTransform()

    datasets = {}
    datasets['train'] = SeqDataset_v2(params.dataset_folder, params.train_file, 
                tile_size=params.tile_size, image_size=params.image_size,
                use_cloud=params.use_cloud, use_polar=params.use_polar,
                image_transform=image_train_transform, tile_transform=tile_train_transform,
                cloud_transform=cloud_train_transform)

    datasets['val'] = SeqDataset_v2(params.dataset_folder, params.val_file, 
            tile_size=params.tile_size, image_size=params.image_size,
            use_cloud=params.use_cloud, use_polar=params.use_polar,
            image_transform=image_val_transform, tile_transform=tile_val_transform,
            cloud_transform=cloud_val_transform)

    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers
    if params.log:
        print('Using {} dataloader workers every process'.format(nw))

    dataloaders = {}
    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(datasets['train'], num_replicas=params.world_size, rank=params.rank, shuffle=True)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, params.batch_size, drop_last=True)
        train_collate_fn = make_collate_fn(datasets['train'])
        dataloaders['train'] = DataLoader(datasets['train'], batch_sampler=train_batch_sampler, num_workers=nw, pin_memory=True, collate_fn=train_collate_fn)
    else:
        train_sampler = None
        train_batch_sampler = None
        train_collate_fn = make_collate_fn(datasets['train'])
        dataloaders['train'] = DataLoader(datasets['train'], batch_sampler=train_batch_sampler, batch_size=params.batch_size, num_workers=nw, pin_memory=True, shuffle=True, drop_last=True, collate_fn=train_collate_fn)

    if params.val_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(datasets['val'], num_replicas=params.world_size, rank=params.rank, shuffle=False)
        val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, params.val_batch_size, drop_last=False)
        val_collate_fn = make_collate_fn(datasets['val'])
        dataloaders['val'] = DataLoader(datasets['val'], sampler=val_batch_sampler, num_workers=nw, pin_memory=True, collate_fn=val_collate_fn)           
    else:
        val_batch_sampler = None
        val_collate_fn = make_collate_fn(datasets['val'])
        dataloaders['val'] = DataLoader(datasets['val'], sampler=val_batch_sampler, batch_size=params.val_batch_size, num_workers=nw, pin_memory=True, shuffle=False, drop_last=False, collate_fn=val_collate_fn)

    do_train(dataloaders, train_sampler, params, use_amp=params.use_amp)


