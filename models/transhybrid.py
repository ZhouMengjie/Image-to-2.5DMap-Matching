""" This implementation is modified from https://github.com/yanghongji2007/cross_view_localization_L2LTR """
import torch.nn as nn
import numpy as np
import ml_collections
from layers.model_crossattn import VisionTransformer

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (8, 32)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config

class TransHybrid(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, out_channels, img_size, data_form):
        """
        dim: feature dimension (default: 512)
        """
        super(TransHybrid, self).__init__()
        self.dim = out_channels
        self.data_form = data_form

        if data_form == 'pano':
            self.size = [img_size, img_size*2]
        elif data_form == 'tile':
            self.size = [img_size, img_size]

        # create the encoders
        config = get_r50_b16_config()
        self.net = VisionTransformer(config, self.size)
        self.net.load_from(np.load('R50+ViT-B_16.npz'))

    def forward(self, batch):
        if self.data_form == 'pano':
            x = batch['images']
        elif self.data_form == 'tile':
            x = batch['tiles']
        return self.net(x), None



