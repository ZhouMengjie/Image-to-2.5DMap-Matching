import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.video_resnet import r3d_18

class VideoModel(nn.Module):
    def __init__(self, params):
        super(VideoModel, self).__init__()
        self.share = params.share
        if params.share:
            self.v_encoder = r3d_18(pretrained=True)
        else:
            self.v_encoder1 = r3d_18(pretrained=True)
            self.v_encoder2 = r3d_18(pretrained=True)
        

    def forward(self, batch):
        if self.share:
            pano_feat = self.v_encoder(batch['images'])
            map_feat = self.v_encoder(batch['tiles'])
        else:
            pano_feat = self.v_encoder1(batch['images'])
            map_feat = self.v_encoder2(batch['tiles'])
        return pano_feat, map_feat



        


        
    


        

