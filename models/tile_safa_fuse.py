import os
import torch
import torch.nn as nn
from torch.nn import init
from layers.resnet_nets import resnet18


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
     
    def forward(self, batch):
        x = batch['tiles']        
        feature = self.resnet(x)
        return feature, None 




