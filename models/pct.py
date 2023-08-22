""" This implementation is modified from https://github.com/qq456cvb/Point-Transformers """
import torch
import torch.nn as nn
from einops import rearrange
from models.pt_utils import farthest_point_sample, index_points, square_distance


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # b, npoint

    new_xyz = index_points(xyz, fps_idx) # b, npoint, 3
    new_points = index_points(points, fps_idx) # b, npoint, c

    dists = square_distance(new_xyz, xyz)  # b, npoint, n
    idx = dists.argsort()[:, :, :nsample]  # b, npoint, nsample

    grouped_points = index_points(points, idx) # b, npoint, nsample, c
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1) # b, npoint, nsample, c
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1) # b, npoint, nsample, 2*c
    return new_xyz, new_points


class PCT(nn.Module):
    def __init__(self, out_channel=256, in_channel=3, npoint=1024, nneighbor=32):
        super().__init__()
        self.npoint = npoint
        self.nsample = nneighbor
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention(channels=256)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(256*5, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        # projection module
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, out_channel, bias=True)

    def forward(self, batch):
        # [batch size, channel, npoint]
        x = batch['coords'] # b, c, n
        xyz = x[:, :3, :] # b, 3, n
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # b, c, n
        x = self.relu(self.bn2(self.conv2(x))) 
        x = x.permute(0, 2, 1) # b, n, c
        xyz = xyz.permute(0, 2, 1)

        l1_xyz, l1_points = sample_and_group(npoint=512, nsample=self.nsample, xyz=xyz, points=x)         
        l1_points = self.gather_local_0(l1_points) # b, c, n
        l1_points = l1_points.permute(0, 2, 1) # b, n, c
        
        l2_xyz, l2_points = sample_and_group(npoint=256, nsample=self.nsample, xyz=l1_xyz, points=l1_points) 
        l2_points = self.gather_local_1(l2_points) # b, c, n

        x = self.pt_last(l2_points)       
        x = torch.cat([x, l2_points], dim=1)
        feature = self.conv_fuse(x) # b, c, n
        # projection module
        x = torch.max(feature, 2)[0]
        x = x.view(batch_size, -1)
        x = self.relu(self.bn6(self.linear1(x))) 
        x = self.dp1(x)
        x = self.linear2(x)
        return x, feature


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # b, npoint, nsample, c 
        x = x.permute(0, 1, 3, 2) # b, npoint, c, nsample
        x = x.reshape(-1, d, s) # b*npoint, c, nsample
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # b*npoint, c, nsample
        x = self.relu(self.bn2(self.conv2(x))) # b*npoint, c, nsample
        x = torch.max(x, 2)[0] # b*npoint, c
        x = x.view(batch_size, -1) # b*npoint, c
        x = x.reshape(b, n, -1).permute(0, 2, 1) # b, c, npoint
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size() # b, c, npoint

        x = self.relu(self.bn1(self.conv1(x))) # b, c, npoint
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # 比较传统的scalar attention, 两点不同：scaling的位置，offset操作
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x) # b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) # change the position of scaling and softmax, suppress the noise
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # offset-attention
        x = x + x_r
        return x

    

