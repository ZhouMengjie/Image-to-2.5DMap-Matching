# The main code structure is modified from https://github.com/jac99/MinkLocMultimodal
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
import numpy as np
import random
from training_pickle.trainer_sam import do_train
from config.utils import Params, get_datetime
from data.dataset_utils_pickle import make_dataloaders
from training_pickle.distributed_utils import init_distributed_mode

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
    parser.add_argument('--config', type=str, required=False, help='Path to configuration file')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=24, required=False, help='Testing batch size')
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

    parser.set_defaults(config='config/config_train.txt')
    parser.set_defaults(train_file='trainstreetlearnU_cmu5kU')
    parser.set_defaults(val_file='hudsonriver5kU')
    parser.set_defaults(eval_files='unionsquare5kU,wallstreet5kU')
    parser.set_defaults(model3d='dgcnn')
    parser.set_defaults(model2d_tile='resnet_safa')
    parser.set_defaults(model2d_pano='resnet_safa')
    parser.set_defaults(loss='MultiInfoNCELoss')
    parser.set_defaults(fuse='2to3')
    parser.set_defaults(optimizer='ASAM')

    args = parser.parse_args()
    seed_all(args.seed)
    
    savedStdout = sys.stdout
    s = get_datetime()
    print_log = open(os.path.join('arun_log',s+'.txt'),'w')
    sys.stdout = print_log

    params = Params(args)

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
        print('Amp mode: {}'.format(args.use_amp))
        params.print()

    dataloaders, train_sampler = make_dataloaders(params)
    do_train(dataloaders, train_sampler, params, use_amp=args.use_amp)
