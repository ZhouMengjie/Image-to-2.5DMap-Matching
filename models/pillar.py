import torch
from torch import nn as nn
import numpy as np
from mmdet3d.registry import MODELS

class Pillar(nn.Module):
    def __init__(self, out_channel=256, in_channel=3):
        super(Pillar, self).__init__()
        pillar_feature_net_cfg = dict(
            type='PillarFeatureNet',
            in_channels=in_channel,
            feat_channels=[64],
            with_distance=True,
            with_cluster_center=True,
            with_voxel_center=False,
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))
        self.pillar_feature_net = MODELS.build(pillar_feature_net_cfg)

        pillar_scatter_cfg = dict(
            type='PointPillarsScatter',
            in_channels=64,
            output_shape=[152, 152])
        self.pillar_scatter = MODELS.build(pillar_scatter_cfg)
        
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, batch):
        B, _, _, _ = batch['tiles'].shape
        features = batch['features']
        voxel_coords = batch['coords']
        npts_per_voxel = batch['npts_per_voxel']
        voxel_features = self.pillar_feature_net(features, npts_per_voxel, voxel_coords)
        bev_features = self.pillar_scatter(voxel_features, voxel_coords, B)
        # sparser encoder or cnn
        x = self.conv1(bev_features)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = nn.functional.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        return x, bev_features, None, None

