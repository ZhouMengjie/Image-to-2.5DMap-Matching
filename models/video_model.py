import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.video_resnet import r3d_18

class VideoModel(nn.Module):
    def __init__(self, params):
        super(VideoModel, self).__init__()
        self.share = params.share
        model_file = 'r3d_18-b3b3357e.pth'
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        if params.share:
            self.v_encoder = r3d_18()
            self.v_encoder.load_state_dict(checkpoint, strict=False)
        else:
            self.v_encoder1 = r3d_18()
            self.v_encoder2 = r3d_18()
            self.v_encoder1.load_state_dict(checkpoint, strict=False)
            self.v_encoder2.load_state_dict(checkpoint, strict=False)
        
    def forward(self, batch):
        images = batch['images'].permute(0,2,1,3,4)
        tiles = batch['tiles'].permute(0,2,1,3,4)
        if self.share:
            pano_feat = self.v_encoder(images)
            map_feat = self.v_encoder(tiles)
        else:
            pano_feat = self.v_encoder1(images)
            map_feat = self.v_encoder2(tiles)
        return pano_feat, map_feat



        


        
    


        

