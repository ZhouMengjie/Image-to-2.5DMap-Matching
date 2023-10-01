import os
import configparser
import time

class ModelParams:
    def __init__(self, img_size, tile_size, args):
        self.img_size = img_size
        self.tile_size = tile_size
        self.model3d = args.model3d
        self.model2d_tile = args.model2d_tile
        self.model2d_pano = args.model2d_pano

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class Params:
    def __init__(self, args):
        assert os.path.exists(args.config), 'Cannot find configuration file: {}'.format(args.config)
        self.params_path = args.config
        self.distributed = args.distributed
        self.val_distributed = args.val_distributed
        self.syncBN = args.syncBN
        self.freeze = args.freeze
        self.port = args.port
        self.load_weights = args.weights  
        self.epoch = args.epoch      
        self.pc_normalize = args.pc_normalize
        self.npoints = args.npoints
        self.nneighbor = args.nneighbor
        self.fuse = args.fuse
        self.fc = args.fc

        config = configparser.ConfigParser()
        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')
        self.use_cloud = params.getboolean('use_cloud', True)
        self.use_rgb = params.getboolean('use_rgb', True)
        self.use_tile = params.getboolean('use_tile', False)
        self.use_snap = params.getboolean('use_snap', False)
        self.use_polar = args.use_polar
        self.use_feat = args.use_feat
        self.img_size = params.getint('img_size', 224) 
        self.tile_size = params.getint('tile_size', 224) 
        if self.use_feat:
            self.feat_size = 24
        else:
            self.feat_size = 1
        self.feat_dim = args.feat_dim

        params = config['TRAIN']
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = 0
                self.t_max = args.epochs
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            elif self.scheduler == 'LambdaLR':
                self.scheduler_milestones = params.getint('scheduler_milestones',5)     
            elif self.scheduler == 'ExpLR':
                self.exp_gamma =  params.getfloat('exp_gamma', 0.99)     
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = args.epochs
        self.weight_decay = args.wd
        self.normalize_embeddings = params.getboolean('normalize_embeddings', True)    # Normalize embeddings during training and evaluation
        
        self.loss = args.loss
        if 'Multi' in self.loss:
            # Weights of different loss component
            weights = params.get('weights', '.3, .3, .3')
            self.weights = [float(e) for e in weights.split(',')]

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = args.margin
        elif 'InfoNCE' in self.loss:
            self.margin = args.margin
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)
        self.train_file = args.train_file

        params = config['Val']
        self.use_val = params.getboolean('use_val', False)
        self.val_file = args.val_file
        
        eval_files = args.eval_files
        self.eval_files = [e for e in eval_files.split(',')]

        # Read model parameters
        self.model_params = ModelParams(self.img_size, self.tile_size, args)

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e not in ['model_params']:
                print('{}: {}'.format(e, param_dict[e]))

        if self.model_params is not None:
            self.model_params.print()
        print('')

