import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tnn.helpers import get_activation_fn, get_norm_fn
from tnn.tno import Tno
from tnn.glu import GLU
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
        x = x.permute(0, 2, 1) # b, n, c
        feature, _ = self.backbone(x) # b, n, c
        # projection module
        x = F.relu(self.bn1(self.linear1(feature.mean(1)))) # global average pooling
        x = self.dp1(x)
        x = self.linear2(x)
        return x, feature


class Backbone(nn.Module):
    def __init__(self, in_channel=3, transformer_dim=512, npoint=4096, nneighbor=16, nblocks=3):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        # self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)   
        self.transformer1 = GTUBlock(embed_dim=32,num_heads=1,expand_ratio=2,rpe_layers=1)
        # self.transformer1 = GTUBlock2D(nneighbor,embed_dim=32,num_heads=1,expand_ratio=2,rpe_layers=1)

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoint // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            # self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
            self.transformers.append(GTUBlock(embed_dim=channel,num_heads=1,expand_ratio=2,rpe_layers=1))
            # self.transformers.append(GTUBlock2D(nneighbor,embed_dim=channel,num_heads=1,expand_ratio=2,rpe_layers=1))
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
        # 重点在这里，使用vector attention，并且引入postion encoding #
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x f
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc) # b x n x f

        # -----------  skip connection -------- #
        res = self.fc2(res) + pre
        return res, attn


class GTUBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, act_fun="silu", causal=False, 
                expand_ratio=3, resi_param=False, use_norm=True, norm_type="simplermsnorm",
                use_decay=False, use_multi_decay=False, rpe_layers=3, rpe_embedding=512, 
                rpe_act="relu", normalize=False, par_type=1, residual=False, gamma=0.99, act_type="none"):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        self.num_heads = num_heads
        self.normalize = normalize
        rpe_embedding = int(min(embed_dim / 8, 32))        
        
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(embed_dim))

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep = Tno(h=num_heads, dim=self.head_dim, rpe_dim=rpe_embedding, causal=causal, 
                        use_decay=use_decay, use_multi_decay=use_multi_decay, residual=residual,
                        act=rpe_act, par_type=par_type, gamma=gamma, bias=bias, act_type=act_type,
                        layers=rpe_layers, norm_type=norm_type)

        # glu
        self.glu = GLU(embed_dim, d1, act_fun, fina_act="None", dropout=0.0, bias=True)

        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)

        # 不同点：trans_conv参考pct
        # self.norm = get_norm_fn(self.norm_type)(d1)        
        # self.trans_conv = nn.Linear(embed_dim, embed_dim, 1)
    
    def forward(self, xyz, x):
        # b, n, d -> b, n, d
        # x: b, h, w, d
        num_heads = self.num_heads

        x, shortcut = self.norm(x), x

        u = self.act(self.u_proj(x)) # b, n, d1
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads) # b, h, n, d1
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> b n (h d)') # b, n, h*d1
        output = u * output
        output = self.o(output) # b, n, d

        x = output + shortcut

        x = self.glu(x)
        
        # 不同点：参考pct, 该操作在pct里被成为offset-attention
        # 文章说该操作sharpens the attention weights and reduce the influence of noise
        # pct的norm使用的是batchnorm
        # x_r = self.act(self.norm(self.trans_conv(x - output))) # offset-attention
        # x = x + x_r  
              
        return x, output

class GTUBlock2D(nn.Module):
    def __init__(self, nneighbor, embed_dim, num_heads, bias=True, act_fun="silu", causal=False, 
                expand_ratio=3, resi_param=False, use_norm=True, norm_type="simplermsnorm",
                use_decay=False, use_multi_decay=False, rpe_layers=3, rpe_embedding=512, 
                rpe_act="relu", normalize=False, par_type=1, residual=False, gamma=0.99, act_type="none"):
        super().__init__()
        self.k = nneighbor
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        self.num_heads = num_heads
        self.normalize = normalize
        rpe_embedding = int(min(embed_dim / 8, 32))        
        
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(embed_dim))

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep1 = Tno(h=num_heads, dim=self.head_dim, rpe_dim=rpe_embedding, causal=causal, 
                        use_decay=use_decay, use_multi_decay=use_multi_decay, residual=residual,
                        act=rpe_act, par_type=par_type, gamma=gamma, bias=bias, act_type=act_type,
                        layers=rpe_layers, norm_type=norm_type)
        # self.toep2 = Tno(h=num_heads, dim=self.head_dim, rpe_dim=rpe_embedding, causal=causal, 
        #         use_decay=use_decay, use_multi_decay=use_multi_decay, residual=residual,
        #         act=rpe_act, par_type=par_type, gamma=gamma, bias=bias, act_type=act_type,
        #         layers=rpe_layers, norm_type=norm_type)
        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)
    
    def forward(self, xyz, x):
        # xyz -> b, n, 3
        # x -> b, n, d
        num_heads = self.num_heads

        shortcut = x

        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b, n, k
        x = index_points(x, knn_idx) # b, n, k, d

        x = self.norm(x) # prenorm

        u = self.act(self.u_proj(x)) # b, n, k, d1
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n m (h d) -> b h n m d', h=num_heads)
        # output = self.toep2(v, dim=-3, normalize=self.normalize) + self.toep1(v, dim=-2, normalize=self.normalize)
        output = self.toep1(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n m d -> b n m (h d)')

        output = u * output # b, n, k, d
        output = torch.einsum('bmnf->bmf', output) # b, n, d
        x = self.o(output) + shortcut
                      
        return x, output


    