# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse
import torch
import scipy.io as sio
import random
import time

from sklearn.neighbors import KDTree
from config.utils import MinkLocParams, get_datetime
from models.model_factory import model_factory
from torch.utils.data import DataLoader
from data.augmentation import ValTransform, ValRGBTransform, ValTileTransform 
from data.streetlearn_no_mc import StreetLearnDataset
from data.dataset_utils_pickle import make_collate_fn_torch
from sklearn.decomposition import PCA

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


def evaluate(model, device, params, exp_name):
    stats = {}
    datasets = {}
    dataloaders = {}
    cloud_transform = ValTransform()
    img_transform = ValRGBTransform()
    tile_transform = ValTileTransform(params.model_params.tile_size)

    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])  # number of workers

    model.eval()

    for location_name in params.eval_files:
        # Extract location name from query and database files
        datasets[location_name] = StreetLearnDataset(params.dataset_folder, location_name,
                                                    transform=cloud_transform,
                                                    image_size=params.model_params.img_size, image_transform=img_transform,
                                                    tile_size=params.model_params.tile_size, tile_transform=tile_transform,
                                                    use_cloud=params.use_cloud, use_rgb=params.use_rgb, use_tile=params.use_tile, 
                                                    use_feat=params.use_feat, use_polar=params.use_polar, 
                                                    normalize=params.pc_normalize, npoints=params.npoints)

        # datasets[location_name].transform.aug_mode = 0 
        collate_fn = make_collate_fn_torch(datasets[location_name])
        dataloaders[location_name] = DataLoader(datasets[location_name], batch_size=params.val_batch_size, collate_fn=collate_fn,
                                       num_workers=nw, pin_memory=True, shuffle=False, drop_last=False)

        count = 0
        similarity = []
        one_percent_recall = []

        database_embeddings = []
        query_embeddings = []

        summary_time = 0
        summary_memory = 0
        for batch in dataloaders[location_name]:
            with torch.no_grad():
                batch = {e: batch[e].to(device) for e in batch}                
                x, map_fmap, pano_fmap = model(batch)
                tile_embedding = x['tile_embedding']
                
                map_fmap = map_fmap.detach().cpu().numpy()
                pano_fmap = pano_fmap.detach().cpu().numpy()
                tile_embedding = tile_embedding.detach().cpu().numpy()

                vis = {}
                vis['map'] = map_fmap
                vis['pano'] = pano_fmap
                vis['tile'] = tile_embedding
                sio.savemat(os.path.join('results', 'vis_2d_pol.mat'), vis)
                np.save(os.path.join('results', ('labels.npy')), batch['labels'].cpu().numpy())
                sys.exit()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=False, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=False, help='Path to the model-specific configuration file')    
    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
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
    parser.add_argument('--model2d_tile', type=str, required=False, help='Model 2D for Tile')  
    parser.add_argument('--model2d_pano', type=str, required=False, help='Model 2D for Pano')     
    parser.add_argument('--use_feat', dest='use_feat', action='store_true')
    parser.add_argument('--use_polar', dest='use_polar', action='store_true')
    parser.add_argument('--feat_dim',type=int, required=False, default=128, help='Feature dimension')
    parser.add_argument('--loss', type=str, required=False, help='Loss')
    parser.add_argument('--margin', type=float, default=0.1, required=False, help='Loss margin')
    parser.add_argument('--npoints', type=int, default=1024, required=False, help='Number of points')
    parser.add_argument('--nneighbor', type=int, default=8, required=False, help='Number of neighbors')
    parser.add_argument('--fuse', type=str, required=False, help='Feature fusion method')
    parser.add_argument('--fc', type=str, required=False, help='Final block')
    parser.add_argument('--optimizer', type=str, required=False, help='Optimizer')
    parser.add_argument('--wd', type=float, default=1e-4, required=False, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=60, required=False, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=4e-5, required=False, help='Initial learning rate')
    parser.add_argument('--scheduler', type=str, required=False, help='LR Scheduler')
    parser.add_argument('--use_regu', dest='use_regu', action='store_true')
    parser.add_argument('--device', type=str, required=False, help='Device')  
    parser.add_argument('--exp_name', type=str, required=False, help='Experiment name')  
    
    parser.set_defaults(config='config/config_train.txt')
    parser.set_defaults(model_config='models/minklocmultimodal.txt')
    parser.set_defaults(train_file='trainstreetlearnU_cmu5kU')
    parser.set_defaults(val_file='hudsonriver5kU')
    parser.set_defaults(eval_files='hudsonriver5kU,unionsquare5kU,wallstreet5kU')
    parser.set_defaults(model3d='dgcnn')
    parser.set_defaults(model2d_tile='resnet')
    parser.set_defaults(model2d_pano='resnet')
    parser.set_defaults(loss='MultiInfoNCELoss')
    parser.set_defaults(fuse='concat')
    parser.set_defaults(optimizer='Adam')
    parser.set_defaults(device='cuda')
    

    args = parser.parse_args()
    seed_all(args.seed)

    savedStdout = sys.stdout
    s = get_datetime()
    print_log = open(os.path.join('test_logs',s+'.txt'),'w')
    sys.stdout = print_log

    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print('Weights: {}'.format(w))
    print('')

    params = MinkLocParams(args)
    params.print()

    device = args.device
    print('Device: {}'.format(device))

    model = model_factory(params)
    model.to(device)
    model_size = torch.cuda.memory_allocated(device=device)
    print('Model size: ', model_size / 1024**2, 'MB')

    if args.weights is not None:
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
            # new_k = new_k.replace('mlps.', 'interp.mlps.') if 'mlps' in new_k else new_k
            state_dict[new_k] = v
        model.load_state_dict(state_dict, strict=True) 
        print('load pretrained {} model!'.format(params.load_weights))

    stats = evaluate(model, device, params, args.exp_name)  
    print_eval_stats(stats)
    
    # Append key experimental metrics to experiment summary file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(params.load_weights)[1]
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    export_eval_stats("experiment_results.txt", prefix, stats)

