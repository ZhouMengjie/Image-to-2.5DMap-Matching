""" Refer to the https://github.com/MCG-NJU/CamLiFlow for the implementation of The Conv2dNormRelu, Conv1dNormRelu and FusionAwareInterp """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.safa import SAFA
from models.utils import batch_indexing, projection, mesh_grid, k_nearest_neighbor, grid_sample_wrapper

class Multimodal(torch.nn.Module):
    def __init__(self, cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size,
                 output_dim: int, fuse_method: str = 'concat',
                 dropout_p: float = None, final_block: str = None):
        super().__init__()

        assert cloud_fe is not None or image_fe is not None or tile_fe is not None

        self.cloud_fe = cloud_fe
        if cloud_fe is None:
            self.cloud_fe_size = 0
        else:
            self.cloud_fe_size = cloud_fe_size

        self.image_fe = image_fe
        if image_fe is None:
            self.image_fe_size = 0
        else:
            self.image_fe_size = image_fe_size

        self.tile_fe = tile_fe
        if tile_fe is None:
            self.tile_fe_size = 0
        else:
            self.tile_fe_size = tile_fe_size

        self.output_dim = output_dim
        self.final_block = final_block
        self.dropout_p = dropout_p
        self.fuse_method = fuse_method

        if self.dropout_p is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout_layer = None

        if fuse_method == 'concat':
            self.fused_dim = self.tile_fe_size + self.cloud_fe_size
        elif fuse_method == 'add':
            assert self.tile_fe_size == self.cloud_fe_size
            self.fused_dim = self.tile_fe_size
        elif fuse_method == '3to2': # 3D to 2D
            self.fused_dim = self.tile_fe_size + self.cloud_fe_size
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.interp = FusionAwareInterp(self.cloud_fe_size, k=4, norm='batch_norm')
            self.fuser = Conv2dNormRelu(self.fused_dim, self.tile_fe_size)
            self.safa = SAFA(28*28, num=8) # use_polar is False
        elif fuse_method == '2to3': # 2D to 3D
            self.fused_dim = self.tile_fe_size + self.cloud_fe_size
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.mlps = nn.Sequential(
                Conv1dNormRelu(self.tile_fe_size, self.tile_fe_size, norm='batch_norm'),
                Conv1dNormRelu(self.tile_fe_size, self.tile_fe_size, norm='batch_norm'),
                Conv1dNormRelu(self.tile_fe_size, self.tile_fe_size, norm='batch_norm'))
            self.fuser = Conv1dNormRelu(self.fused_dim, self.fused_dim*4)
        else:
            raise NotImplementedError('Unsupported fuse method: {}'.format(self.fuse_method))

        if self.final_block is None:
            self.final_net = None
        elif self.final_block == 'fc':
            self.final_net = nn.Linear(self.fused_dim, output_dim)
        elif self.final_block == 'mlp':
            temp_channels = self.output_dim
            self.final_net = nn.Sequential(nn.Linear(self.fused_dim, temp_channels, bias=False),
                                           nn.BatchNorm1d(temp_channels),
                                           nn.ReLU(inplace=True), 
                                           nn.Dropout(),
                                           nn.Linear(temp_channels, output_dim))
        else:
            raise NotImplementedError('Unsupported final block: {}'.format(self.final_block))             

    def forward(self, batch):
        y = {}
        if self.image_fe is not None:
            image_embedding, _ = self.image_fe(batch)
            y['image_embedding'] = image_embedding

        if self.cloud_fe is not None:
            cloud_embedding, _, xyz, center = self.cloud_fe(batch)
            y['cloud_embedding'] = cloud_embedding

        if self.tile_fe is not None:
            tile_embedding, _ = self.tile_fe(batch)
            y['tile_embedding'] = tile_embedding

        if self.cloud_fe is not None and self.tile_fe is not None:
            assert cloud_embedding.shape[0] == tile_embedding.shape[0]
            if self.fuse_method == 'concat':
                x = torch.cat([cloud_embedding, tile_embedding], dim=1)
            elif self.fuse_method == 'add':
                assert cloud_embedding.shape == tile_embedding.shape
                x = cloud_embedding + tile_embedding         
            elif self.fuse_method == '3to2':                       
                tile_embedding = self.up(tile_embedding)
                batch_size, _, image_h, image_w = tile_embedding.shape
                npoints = cloud_embedding.shape[2]
                sensor_h, sensor_w = 152, 152
                image = projection(center,xyz,image_h,image_w,sensor_h,sensor_w,npoints,batch_size) 
                # 3d to 2d
                out = self.interp(image, tile_embedding, cloud_embedding)
                # concatenation and fusion
                out = self.fuser(torch.cat([out, tile_embedding], dim=1))
                f = out.view(batch_size, -1, image_h*image_w)
                w = self.safa(f) 
                x = torch.matmul(f, w).view(batch_size, -1)   
            elif self.fuse_method == '2to3':      
                tile_embedding = self.up(tile_embedding)
                batch_size, _, image_h, image_w = tile_embedding.shape
                npoints = cloud_embedding.shape[2]
                sensor_h, sensor_w = 152, 152
                image = projection(center,xyz,image_h,image_w,sensor_h,sensor_w,npoints,batch_size) 
                # 2d to 3d                
                feat_2d_to_3d = grid_sample_wrapper(tile_embedding, image)
                out = self.mlps(feat_2d_to_3d)
                # concatenation and fusion
                out = self.fuser(torch.cat([out, cloud_embedding], dim=1)) 
                x = F.adaptive_max_pool1d(out, 1).view(batch_size, -1) 
            else:
                raise NotImplementedError('Unsupported fuse method: {}'.format(self.fuse_method))
        elif self.cloud_fe is not None:
            x = cloud_embedding
        elif self.tile_fe is not None:
            x = tile_embedding
        else:
            raise NotImplementedError('Unsupported cross modality')

        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        if self.final_net is not None:
            x = self.final_net(x)
            
        # x is (batch_size, output_dim) tensor
        y['embedding'] = x
        return y

    def print_info(self):
        print('Model class: Multimodal')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        if self.cloud_fe is not None:
            n_params = sum([param.nelement() for param in self.cloud_fe.parameters()])
            print('Cloud feature extractor parameters: {}'.format(n_params))

        if self.image_fe is not None:
            n_params = sum([param.nelement() for param in self.image_fe.parameters()])
            print('Image feature extractor parameters: {}'.format(n_params))

        if self.tile_fe is not None:
            n_params = sum([param.nelement() for param in self.tile_fe.parameters()])
            print('Tile feature extractor parameters: {}'.format(n_params))

        print('Fuse method: {}'.format(self.fuse_method))
        if self.fuse_method == '3to2':
            n_params_mlps = sum([param.nelement() for param in self.interp.parameters()])
            n_params_fuser = sum([param.nelement() for param in self.fuser.parameters()])
            n_params_safa = sum([param.nelement() for param in self.safa.parameters()])
            print('Fusion block parameters: {}'.format(n_params_mlps+n_params_fuser+n_params_safa))
        elif self.fuse_method == '2to3':
            n_params_mlps = sum([param.nelement() for param in self.mlps.parameters()])
            n_params_fuser = sum([param.nelement() for param in self.fuser.parameters()])
            print('Fusion block parameters: {}'.format(n_params_mlps+n_params_fuser))
  

        if self.dropout_p is not None:
            print('Dropout p: {}'.format(self.dropout_p))

        print('Final block: {}'.format(self.final_block))
        if self.final_net is not None:
            n_params = sum([param.nelement() for param in self.final_net.parameters()])
            print('FC block parameters: {}'.format(n_params))

        print('Dimensionality of cloud features: {}'.format(self.cloud_fe_size))
        print('Dimensionality of image features: {}'.format(self.image_fe_size))
        print('Dimensionality of tile features: {}'.format(self.tile_fe_size))
        print('Dimensionality of final descriptor: {}'.format(self.output_dim))


class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, activation='leaky_relu'):
        super().__init__()
        self.conv_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.relu_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.relu_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.relu_fn(x)
        return x


class Conv1dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, activation='leaky_relu'):
        super().__init__()
        self.conv_fn = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.relu_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.relu_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.relu_fn(x)
        return x


class FusionAwareInterp(nn.Module):
    def __init__(self, n_channels_3d, k=3, norm=None) -> None:
        super().__init__()
        self.k = k
        self.mlps = nn.Sequential(
            Conv2dNormRelu(n_channels_3d + 3, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
        )

    def forward(self, uv, feat_2d, feat_3d):
        bs, _, h, w = feat_2d.shape

        grid = mesh_grid(bs, h, w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        nn_indices = k_nearest_neighbor(uv, grid, k=self.k)[..., 0]  # [B, HW]
        nn_feat2d = batch_indexing(grid_sample_wrapper(feat_2d, uv), nn_indices)  # [B, n_channels_2d, HW]
        nn_feat3d = batch_indexing(feat_3d, nn_indices)  # [B, n_channels_3d, HW]
        nn_offset = batch_indexing(uv, nn_indices) - grid  # [B, 2, HW]
        nn_corr = torch.mean(nn_feat2d * feat_2d.reshape(bs, -1, h * w), dim=1, keepdim=True)  # [B, 1, HW]

        feat = torch.cat([nn_offset, nn_corr, nn_feat3d], dim=1)  # [B, n_channels_3d + 3, HW]
        feat = feat.reshape([bs, -1, h, w])  # [B, n_channels_3d + 3, H, W]
        final = self.mlps(feat)  # [B, n_channels_3d, H, W]

        return final

