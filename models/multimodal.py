# Author: Jacek Komorowski
# Warsaw University of Technology

# Model processing LiDAR point clouds and RGB images
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
from layers.safa import SAFA
CE = torch.nn.BCELoss(reduction='sum')
from models.utils import batch_indexing, projection, mesh_grid, k_nearest_neighbor, grid_sample_wrapper
from layers.fusion_v3 import Fusion

class Multimodal(torch.nn.Module):
    def __init__(self, cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size,
                 output_dim: int, fuse_method: str = 'concat',
                 dropout_p: float = None, final_block: str = None,
                 mm_fusion = False,
                 regularizer = False, latent_size = 6):
        # cloud_fe: cloud feature extractor, returns tensor(batch_size, cloud_fe_size)
        # imaged_fe: image feature extractor, returns tensor(batch_size, image_fe_size)
        # output_dim: dimensionality of the fused global descriptor
        # dropout_p: whether to use Dropout after feature concatenation and before the fully connected block
        # add_fc_block: if True, a fully connected block is added after feature concatenation and optional Dropout block

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
        self.regularizer = regularizer
        self.mm_fusion = mm_fusion

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
            self.interp = FusionAwareInterpCVPR(self.cloud_fe_size, k=4, norm='batch_norm')
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
        elif fuse_method == 'attn':
            self.fused_dim = self.cloud_fe_size
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.fuser = Fusion(in_channels=self.fused_dim)
            self.mlps = Conv1dNormRelu(self.fused_dim, self.fused_dim*8)
        elif fuse_method == 'bev':
            self.fused_dim = self.tile_fe_size + self.cloud_fe_size
            self.fuser = Conv2dNormRelu(self.fused_dim, self.tile_fe_size)
            self.safa = SAFA(7*7, num=8)
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

        if self.regularizer:
            self.fc1_img = nn.Linear(self.tile_fe_size, latent_size)
            self.fc2_img = nn.Linear(self.tile_fe_size, latent_size)
            self.fc1_pc = nn.Linear(self.cloud_fe_size, latent_size)
            self.fc2_pc = nn.Linear(self.cloud_fe_size, latent_size)
            self.tanh = nn.Tanh()

        if self.mm_fusion:
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.fusion_net = Fusion(in_channels=self.image_fe_size)               

    def forward(self, batch):
        y = {}
        if self.image_fe is not None:
            image_embedding, _ = self.image_fe(batch)
            # assert image_embedding.dim() == 2
            # assert image_embedding.shape[1] == self.image_fe_size
            y['image_embedding'] = image_embedding

        if self.cloud_fe is not None:
            cloud_embedding, _, xyz, center = self.cloud_fe(batch)
            # assert cloud_embedding.dim() == 2
            # assert cloud_embedding.shape[1] == self.cloud_fe_size
            y['cloud_embedding'] = cloud_embedding

        if self.tile_fe is not None:
            tile_embedding, _ = self.tile_fe(batch)
            # assert tile_embedding.dim() == 2
            # assert tile_embedding.shape[1] == self.tile_fe_size
            y['tile_embedding'] = tile_embedding

        latent_loss = 0.
        if self.cloud_fe is not None and self.tile_fe is not None:
            assert cloud_embedding.shape[0] == tile_embedding.shape[0]
            if self.fuse_method == 'concat':
                x = torch.cat([cloud_embedding, tile_embedding], dim=1)
            elif self.fuse_method == 'add':
                assert cloud_embedding.shape == tile_embedding.shape
                x = cloud_embedding + tile_embedding         
            elif self.fuse_method == '3to2': 
                # tile_feature_map = tile_embedding.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('tile2.npy')), tile_feature_map)
                # point_feature_map = cloud_embedding.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('point2.npy')), point_feature_map)        
                # xyz_feature_map = xyz.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('xyz2.npy')), xyz_feature_map)                       
                tile_embedding = self.up(tile_embedding)
                batch_size, _, image_h, image_w = tile_embedding.shape
                npoints = cloud_embedding.shape[2]
                sensor_h, sensor_w = 152, 152
                image = projection(center,xyz,image_h,image_w,sensor_h,sensor_w,npoints,batch_size) 
                # 3d to 2d
                out = self.interp(image, tile_embedding, cloud_embedding)
                # point2d_feature_map = out.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('tile3d.npy')), point2d_feature_map)                       
                # concatenation and fusion
                out = self.fuser(torch.cat([out, tile_embedding], dim=1))
                # fuse_feature_map = out.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('fuse2.npy')), fuse_feature_map)                        
                f = out.view(batch_size, -1, image_h*image_w)
                w = self.safa(f) 
                x = torch.matmul(f, w).view(batch_size, -1)   
            elif self.fuse_method == '2to3': 
                # tile_feature_map = tile_embedding.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('tile.npy')), tile_feature_map)
                # point_feature_map = cloud_embedding.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('point.npy')), point_feature_map)        
                # xyz_feature_map = xyz.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('xyz.npy')), xyz_feature_map)        
                tile_embedding = self.up(tile_embedding)
                batch_size, _, image_h, image_w = tile_embedding.shape
                npoints = cloud_embedding.shape[2]
                sensor_h, sensor_w = 152, 152
                image = projection(center,xyz,image_h,image_w,sensor_h,sensor_w,npoints,batch_size) 
                # 2d to 3d                
                feat_2d_to_3d = grid_sample_wrapper(tile_embedding, image)
                out = self.mlps(feat_2d_to_3d)
                # point2d_feature_map = out.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('point2d.npy')), point2d_feature_map)        
                # concatenation and fusion
                out = self.fuser(torch.cat([out, cloud_embedding], dim=1)) 
                # fuse_feature_map = out.cpu().numpy()
                # np.save(os.path.join('results','feature_maps', ('fuse.npy')), fuse_feature_map)        
                x = F.adaptive_max_pool1d(out, 1).view(batch_size, -1) 
            elif self.fuse_method == 'attn':
                tile_embedding = self.up(tile_embedding) 
                tile_embedding, cloud_embedding = self.fuser(tile_embedding, cloud_embedding) 
                x = (tile_embedding + cloud_embedding) / 2
                x = x.unsqueeze(-1)
                x = self.mlps(x)
                x = x.squeeze(-1)
            elif self.fuse_method == 'bev':
                batch_size, _, image_h, image_w = tile_embedding.shape
                out = self.fuser(torch.cat([tile_embedding, cloud_embedding], dim=1))
                f = out.view(batch_size, -1, image_h*image_w)
                w = self.safa(f) 
                x = torch.matmul(f, w).view(batch_size, -1)   
            else:
                raise NotImplementedError('Unsupported fuse method: {}'.format(self.fuse_method))
            
            if self.regularizer:
                batch_size = tile_embedding.shape[0] // 2
                tile_t1_feats = tile_embedding[:batch_size, :]
                tile_t2_feats = tile_embedding[batch_size: , :]
                tile_feats = torch.stack([tile_t1_feats,tile_t2_feats]).mean(dim=0)
                pc_t1_feats = cloud_embedding[:batch_size, :]
                pc_t2_feats = cloud_embedding[batch_size: , :]
                pc_feats = torch.stack([pc_t1_feats,pc_t2_feats]).mean(dim=0)
                mu_rgb = self.fc1_img(tile_feats)
                logvar_rgb = self.fc2_img(tile_feats)
                mu_depth = self.fc1_pc(pc_feats)
                logvar_depth = self.fc2_pc(pc_feats)               
                mu_depth = self.tanh(mu_depth)
                mu_rgb = self.tanh(mu_rgb)
                logvar_depth = self.tanh(logvar_depth)
                logvar_rgb = self.tanh(logvar_rgb)
                z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
                dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
                z_depth = self.reparametrize(mu_depth, logvar_depth)
                dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
                bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
                    self.kl_divergence(dist_depth, dist_rgb))
                z_rgb_norm = torch.sigmoid(z_rgb)
                z_depth_norm = torch.sigmoid(z_depth)
                ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
                ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
                latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
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

        if self.mm_fusion:
            x_ = self.up(image_embedding)
            x_, x = self.fusion_net(image_embedding, cloud_embedding) # or x_, out
            y['image_embedding'] = x_
            
        # assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1],self.output_dim)
        # x is (batch_size, output_dim) tensor
        y['embedding'] = x
        return y, None, None

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
        elif self.fuse_method == '2to3' or self.fuse_method == 'attn':
            n_params_mlps = sum([param.nelement() for param in self.mlps.parameters()])
            n_params_fuser = sum([param.nelement() for param in self.fuser.parameters()])
            print('Fusion block parameters: {}'.format(n_params_mlps+n_params_fuser))
  

        if self.dropout_p is not None:
            print('Dropout p: {}'.format(self.dropout_p))

        print('Final block: {}'.format(self.final_block))
        if self.final_net is not None:
            n_params = sum([param.nelement() for param in self.final_net.parameters()])
            print('FC block parameters: {}'.format(n_params))

        if self.mm_fusion:
            n_params = sum([param.nelement() for param in self.fusion_net.parameters()])
            print('Fusion bottleneck parameters: {}'.format(n_params))

        print('Dimensionality of cloud features: {}'.format(self.cloud_fe_size))
        print('Dimensionality of image features: {}'.format(self.image_fe_size))
        print('Dimensionality of tile features: {}'.format(self.tile_fe_size))
        print('Dimensionality of final descriptor: {}'.format(self.output_dim))

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


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


class FusionAwareInterpCVPR(nn.Module):
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


class FusionAwareInterp(nn.Module):
    def __init__(self, n_channels_3d, k=1, norm=None):
        super().__init__()
        self.k = k
        self.out_conv = Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm)
        self.score_net = nn.Sequential(
            Conv2dNormRelu(3, 16),  # [dx, dy, |dx, dy|_2, sim]
            Conv2dNormRelu(16, n_channels_3d, act='sigmoid'),
        )

    def forward(self, uv, feat_2d, feat_3d):
        bs, _, image_h, image_w = feat_2d.shape
        n_channels_3d = feat_3d.shape[1]

        grid = mesh_grid(bs, image_h, image_w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        knn_indices = k_nearest_neighbor(uv, grid, self.k)  # [B, HW, k]

        knn_uv, knn_feat3d = torch.split(
            batch_indexing(
                torch.cat([uv, feat_3d], dim=1),
                knn_indices
            ), [2, n_channels_3d], dim=1)

        knn_offset = knn_uv - grid[..., None]  # [B, 2, HW, k]
        knn_offset_norm = torch.linalg.norm(knn_offset, dim=1, keepdim=True)  # [B, 1, HW, k]

        score_input = torch.cat([knn_offset, knn_offset_norm], dim=1)  # [B, 4, HW, K]
        score = self.score_net(score_input)  # [B, n_channels_3d, HW, k]
        # score = softmax(score, dim=-1)  # [B, n_channels_3d, HW, k]

        final = score * knn_feat3d  # [B, n_channels_3d, HW, k]
        final = final.sum(dim=-1).reshape(bs, -1, image_h, image_w)  # [B, n_channels_3d, H, W]
        final = self.out_conv(final)

        return final
