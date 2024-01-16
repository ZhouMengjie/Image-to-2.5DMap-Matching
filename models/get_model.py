import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seq_model import TransMixer
from models.seq_model import TransMixer_v2
from models.seq_model import SeqNet
from models.seq_model import SeqNet_v2
from models.seq_model import Delta
from models.seq_model import Smooth
from models.seq_model import Baseline
from models.seq_model import SeqPool
from models.seq_model import SeqPool_v2
from models.seq_model import TransMixerMask


def get_model(params):
    if params.model_type == 'transmixer' and params.share:
        model = TransMixer(params.feat_dim, nHead=params.num_heads, numLayers=params.num_layers, max_length=params.seq_len, pool=params.pool)
    elif params.model_type == 'transmixer' and not params.share:
        model = TransMixer_v2(params.feat_dim, nHead=params.num_heads, numLayers=params.num_layers, max_length=params.seq_len)
    elif params.model_type == 'smoothing':
        model = Smooth()
    elif params.model_type == 'seqnet' and params.share:
        model = SeqNet(params.feat_dim, seqL=params.seq_len, w=params.w, pool=params.pool)
    elif params.model_type == 'seqnet' and not params.share:
        model = SeqNet_v2(params.feat_dim, seqL=params.seq_len, w=params.w)
    elif params.model_type == 'delta':
        model = Delta(params.feat_dim, seqL=params.seq_len)
    elif params.model_type == 'baseline':
        model = Baseline()
    elif params.model_type == 'seqpool':
        model = SeqPool(params.feat_dim)
    elif params.model_type == 'seqpool_v2':
        model = SeqPool_v2(params.feat_dim)
    elif params.model_type == 'transmixer_mask':
        model = TransMixerMask(params.feat_dim, nHead=params.num_heads, numLayers=params.num_layers, max_length=params.seq_len, pool=params.pool, max_masked=params.max_masked, special_mask=params.special_mask)
    else:
        raise NotImplementedError('Unsupported Model Type: {}'.format(params.model_type))
    return model
    


        

