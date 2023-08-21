"This model is the map sub-network used only for testing, not training "
import os
import torch
import torch.nn as nn
from torch.nn import init
from layers.resnet_nets import resnet18
from layers.safa import SAFA
import numpy as np


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class EmbNet(nn.Module):
    def __init__(self, inSize, embedding_dim):
        super(EmbNet, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(inSize, 512)
        self.fc2 = nn.Linear(512,embedding_dim)
        self.bn1 = nn.BatchNorm1d(num_features=inSize)
        self.bn2 = nn.BatchNorm1d(num_features=512)

    def forward(self,x):
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Tile_SAFA(torch.nn.Module):
    def __init__(self, out_channels, tile_size, use_polar=False, sa_num=8):
        super(Tile_SAFA, self).__init__()
        # self.resnet = resnet18(pretrained=True)
        self.resnet = resnet18()
        # Load weights
        model_file = 'resnet18-5c106cde.pth'
        if not os.access(model_file, os.W_OK):
            weight_url = 'https://download.pytorch.org/models/' + model_file
            os.system('wget ' + weight_url)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        self.resnet.load_state_dict(checkpoint, strict=False)
        
        if use_polar:
            in_dim = (tile_size//32)**2*2
        else:
            in_dim = (tile_size//32)**2
        self.safa = SAFA(in_dim, sa_num)   
         
        # in_channels = 512*sa_num    
        # self.embnet = EmbNet(in_channels, out_channels)
        # init_weights(self.embnet)
     
    def forward(self, batch):
        x = batch['tiles']        
        feature = self.resnet(x)
        # tile_feature_map = feature.cpu().numpy()
        # np.save(os.path.join('results','feature_maps', ('tile3.npy')), tile_feature_map)               
        B, C, _, _ = feature.shape
        f = feature.view(B, C, -1)
        w = self.safa(f) 
        x = torch.matmul(f, w).view(B, -1)
        # x = self.embnet(x)
        return x, feature 




