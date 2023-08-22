""" This implementation is modified from https://github.com/qq456cvb/Point-Transformers """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.pt_utils import sample_and_group, sample_and_group_all, square_distance, index_points


class PT2(nn.Module):
    def __init__(self, out_channel=256, in_channel=3, transformer_dim=512, npoint=4096, nneighbor=16, nblocks=3):
        super().__init__()
        self.backbone = Backbone(in_channel, transformer_dim, npoint, nneighbor, nblocks)
        # projection module
        self.linear1 = nn.Linear(32 * 2 ** nblocks, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, out_channel, bias=True)
   
    def forward(self, batch):
        # [batch size, channel, npoint]
        x = batch['coords']  
        xyz = batch['xyz']
        center = batch['center']
        x = x.permute(0, 2, 1) # b, n, c
        feature, _ = self.backbone(x) # b, n, c
        # projection module
        x = F.relu(self.bn1(self.linear1(feature.mean(1)))) # global average pooling
        x = self.dp1(x)
        x = self.linear2(x)
        return x, feature, xyz, center


class Backbone(nn.Module):
    def __init__(self, in_channel=3, transformer_dim=512, npoint=4096, nneighbor=16, nblocks=3):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)   
       
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoint // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        # [batch size, npoint, channel]
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class TransitionDown(nn.Module):
    def __init__(self, npoint, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(npoint, radius=None, nsample=nneighbor, in_channel=channels[0], mlp=channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2) # [B, npoint, C+D]
        return new_xyz, new_points


class TransformerBlock(nn.Module):  
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__() # d_model = transformer_dim
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential( 
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k # number of neighbors
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3
        
        pre = features
        # fc1 ---> transformer ---> fc2 ------- #
        x = self.fc1(features) # b x n x f
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q -> b x n x f,  k/v -> b x n x k x f
        # ----------- position encoding -------- # 
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        # -----------  vector attention -------- # 
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x f
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc) # b x n x f

        # -----------  skip connection -------- #
        res = self.fc2(res) + pre
        return res, attn




    