# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
from cProfile import label
import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
import torch
import scipy.io as sio
import random

from config.utils import MinkLocParams
from models.model_factory import model_factory
from torch.utils.data import DataLoader
from data.augmentation import ValTransform, ValRGBTransform

from data.streetlearn_pickle import StreetLearnDataset
from data.dataset_utils_pickle import make_collate_fn
from training_pickle.distributed_utils import init_distributed_mode

DEBUG = False

def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('Seed: {}'.format(random_seed))


def evaluate(model, device, params):
    stats = {}
    datasets = {}
    dataloaders = {}
    cloud_transform = ValTransform()
    img_transform = ValRGBTransform()

    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers

    model.eval()

    for location_name in params.eval_files:
        # Extract location name from query and database files
        datasets[location_name] = StreetLearnDataset(params.dataset_folder, location_name,
                                           transform=cloud_transform,
                                           image_size=params.model_params.img_size, image_transform=img_transform,
                                           use_cloud=params.use_cloud, use_rgb=params.use_rgb, use_feat=params.use_feat, 
                                           normalize=params.pc_normalize, npoints=params.npoints)
        collate_fn = make_collate_fn(datasets[location_name])  

        if params.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(datasets[location_name], num_replicas=params.world_size, rank=params.rank, shuffle=False)
            batch_sampler = torch.utils.data.BatchSampler(sampler, params.val_batch_size, drop_last=False)
            dataloaders[location_name] = DataLoader(datasets[location_name], sampler=batch_sampler, collate_fn=collate_fn,
                                          num_workers=nw, pin_memory=True)           
        else:
            batch_sampler = None
            dataloaders['val'] = DataLoader(datasets['val'], sampler=batch_sampler, batch_size=params.val_batch_size, collate_fn=collate_fn,
                                num_workers=nw, pin_memory=True, shuffle=False, drop_last=False)

        count = 0
        for batch in dataloaders[location_name]:
            with torch.no_grad():
                batch = {e: batch[e].to(device) for e in batch}
                labels = batch['labels']
                x = model(batch)

                cloud_embedding = x['cloud_embedding']
                image_embedding = x['image_embedding']
                if params.normalize_embeddings:
                    cloud_embedding = torch.nn.functional.normalize(cloud_embedding, p=2, dim=1)  # Normalize embeddings, dim=row
                    image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=1)

                cloud_embedding = cloud_embedding.detach().cpu().numpy()
                image_embedding = image_embedding.detach().cpu().numpy()
                cloud_labels = labels.detach().cpu().numpy()
                image_labels = labels.detach().cpu().numpy()

            torch.cuda.empty_cache()
            count = count + 1
            # check data here: batch idx
            print('Batch[{0}]/Location[{1}]/NOP[{2}]: {3}'.format(count,location_name,batch['coords'].shape[2],batch['labels']))
            
            database_embeddings = {cloud_labels[i]:cloud_embedding[i] for i in range(cloud_labels)}
            query_embeddings = {image_labels[i]:image_embedding[i] for i in range(image_labels)}

        pred = {}
        pred['ref'] = database_embeddings
        pred['qry'] = query_embeddings
        model_name = os.path.split(params.load_weights)[1]
        model_name = model_name.split('.')[0]
        sio.savemat(os.path.join('results', location_name+'_'+model_name+'_'+device+'.mat'), pred)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=False, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=False, help='Path to the model-specific configuration file')    
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Training batch size')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--epoch', type=int, default=1, required=False, help='Initial training epoch')    
    parser.add_argument('--pc_normalize', dest='pc_normalize', action='store_true')
    parser.add_argument('--train_file', type=str, required=False, help='Train file')
    parser.add_argument('--val_file', type=str, required=False, help='Val file')
    parser.add_argument('--eval_files', type=str, required=False, help='Eval files')
    parser.add_argument('--distributed', dest='distributed', action='store_true')
    parser.add_argument('--val_distributed', dest='val_distributed', action='store_true')
    parser.add_argument('--freeze', dest='freeze', action='store_true')
    parser.add_argument('--syncBN', dest='syncBN', action='store_true')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Seed')
    parser.add_argument('--port', type=str, default='12363', required=False, help='Port')  
    parser.add_argument('--mink_quantization_size', type=float, default=0.01, required=False, help='Quantization size')
    parser.add_argument('--model3d', type=str, required=False, help='Model 3D')  
    parser.add_argument('--use_feat', dest='use_feat', action='store_true')
    parser.add_argument('--loss', type=str, required=False, help='Loss')
    parser.add_argument('--margin', type=float, default=0.1, required=False, help='Loss margin')
    parser.add_argument('--npoints', type=int, default=8192, required=False, help='Number of points')
    parser.set_defaults(config='config/config_train.txt')
    parser.set_defaults(model_config='models/minklocmultimodal.txt')
    parser.set_defaults(train_file='trainstreetlearnU_cmu5kU')
    parser.set_defaults(val_file='hudsonriver5kU')
    parser.set_defaults(eval_files='unionsquare5kU,wallstreet5kU')
    parser.set_defaults(weights='weights/minklocmultimodal_baseline.pth')
    parser.set_defaults(model3d='pointnet')
    parser.set_defaults(loss='MultiBatchHardTripletMarginLoss')

    args = parser.parse_args()
    seed_all(args.seed)

    params = MinkLocParams(args)

    if params.distributed:
        init_distributed_mode(params)
    else:
        params.log = True
        if torch.cuda.is_available():
            params.device = torch.device('cuda')
        else:
            params.device = torch.device('cpu')

    
    if params.log:
        print('Config path: {}'.format(args.config))
        print('Model config path: {}'.format(args.model_config))
        if args.weights is None:
            w = 'RANDOM WEIGHTS'
        else:
            w = args.weights
        print('Weights: {}'.format(w))
        params.print()

    device = params.device
    model = model_factory(params)
    model.to(device)

    if params.load_weights is not None:
        assert os.path.exists(params.load_weights), 'Cannot open network weights: {}'.format(params.load_weights)
        checkpoint = torch.load(params.load_weights, map_location=device)     
        state_dict = {}
        try:
            ckp = checkpoint['model']
            op_ckp = checkpoint['optimizer']
        except:
            ckp = checkpoint
            op_ckp = None
        for k, v in ckp.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            if new_k == 'cloud_fe.backbone.conv0.kernel' and v.shape[1] != params.feat_size:
                new_k = 'ignore' 
            state_dict[new_k] = v
        model.load_state_dict(state_dict, strict=False) 
        print('load pretrained {} model!'.format(params.load_weights))

    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.gpu])

    stats = evaluate(model, device, params)  


