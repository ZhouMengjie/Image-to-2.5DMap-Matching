import os
import torch
import torch.nn as nn
from torch.nn import init
from layers.resnet_nets import resnet50

class Pano_SAFA(torch.nn.Module):
    def __init__(self, out_channels, img_size, sa_num=8):
        super(Pano_SAFA, self).__init__()
        self.resnet = resnet50()    
        # Load weights
        model_file = 'resnet50_places365.pth.tar'
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.resnet.load_state_dict(state_dict, strict=False)        

    def forward(self, batch):
        x = batch['images']
        feature = self.resnet(x)        
        return feature, None    




