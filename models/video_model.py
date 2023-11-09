import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.video_resnet import r3d_18, mc3_18, r2plus1d_18
# from timesformer.models.vit import TimeSformer

class VideoModel(nn.Module):
    def __init__(self, params):
        super(VideoModel, self).__init__()
        self.share = params.share
        model_file = 'r2plus1d_18-91a641e6.pth'
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        if params.share:
            self.v_encoder = r2plus1d_18()
            self.v_encoder.load_state_dict(checkpoint, strict=False)
            # self.v_encoder = TimeSformer(img_size=params.image_size,
            #                 num_classes=0,
            #                 num_frames=params.seq_len,
            #                 attention_type='divided_space_time',
            #                 pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')
        else:
            self.v_encoder1 = r2plus1d_18()
            self.v_encoder2 = r2plus1d_18()
            self.v_encoder1.load_state_dict(checkpoint, strict=False)
            self.v_encoder2.load_state_dict(checkpoint, strict=False)
            # self.v_encoder1 = TimeSformer(img_size=params.image_size,
            #                 num_classes=0,
            #                 num_frames=params.seq_len,
            #                 attention_type='divided_space_time',
            #                 pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')
            # self.v_encoder2 = TimeSformer(img_size=params.image_size,
            #                 num_classes=0,
            #                 num_frames=params.seq_len,
            #                 attention_type='divided_space_time',
            #                 pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')
        
    def forward(self, batch):
        images = batch['images'].permute(0,2,1,3,4)  # B, C, S, H, W
        tiles = batch['tiles'].permute(0,2,1,3,4)
        if self.share:
            pano_feat = self.v_encoder(images)
            map_feat = self.v_encoder(tiles)
        else:
            pano_feat = self.v_encoder1(images)
            map_feat = self.v_encoder2(tiles)
        return pano_feat, map_feat



        


        
    


        

