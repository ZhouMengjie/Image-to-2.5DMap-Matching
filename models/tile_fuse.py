"This model is the map sub-network used only for testing, not training "
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from layers.resnet_nets import resnet18


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
    def __init__(self, tile_size, out_channels, use_polar=False):
        super(EmbNet, self).__init__()
        if use_polar:       
            embedding_channel = 512*2
        else:
            embedding_channel = 512
        self.linear1 = nn.Conv2d(embedding_channel, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Conv2d(512, out_channels, kernel_size=1, bias=True)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class Tile(torch.nn.Module):
    def __init__(self, out_channels, tile_size, use_polar=False):
        super(Tile, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.embnet = EmbNet(tile_size, out_channels, use_polar)
        init_weights(self.embnet)
     
    def forward(self, batch):
        x = batch['tiles']        
        feature = self.resnet(x)
        # x = feature.view(feature.size(0),feature.size(1),-1) # b, c, h*w
        x = self.embnet(feature)        
        return x, feature 




