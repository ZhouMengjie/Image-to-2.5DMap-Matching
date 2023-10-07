import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seq_model import TransMixer
from models.seq_model import SeqNet
from models.seq_model import Delta
from models.seq_model import Smooth
from models.seq_model import Baseline


def get_model(params):
    if params.model_type == 'transmixer':
        model = TransMixer(params.feat_dim, nHead=params.num_heads, numLayers=params.num_layers, max_length=params.seq_len)
    elif params.model_type == 'smoothing':
        model = Smooth()
    elif params.model_type == 'seqnet':
        model = SeqNet(params.feat_dim, params.feat_dim, params.seq_len)
    elif params.model_type == 'delta':
        model = Delta(params.feat_dim, params.seq_len)
    elif params.model_type == 'baseline':
        model = Baseline()
    else:
        raise NotImplementedError('Unsupported Model Type: {}'.format(params.model_type))
    return model
    


        

