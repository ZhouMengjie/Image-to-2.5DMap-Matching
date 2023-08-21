import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tnn.helpers import get_activation_fn, get_norm_fn
from tnn.tno import Tno

class PT(nn.Module):
    def __init__(self, out_channel, in_channel, embedding_channel=1024):
        super(PT, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.tnn1 = GTU(embed_dim=64,num_heads=1,expand_ratio=2,rpe_layers=1)
        self.tnn2 = GTU(embed_dim=128,num_heads=1,expand_ratio=2,rpe_layers=1)
        self.tnn3 = GTU(embed_dim=256,num_heads=1,expand_ratio=2,rpe_layers=1)
        self.tnn4 = GTU(embed_dim=512,num_heads=1,expand_ratio=2,rpe_layers=1)

        # projection module
        self.linear1 = nn.Linear(embedding_channel, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, out_channel, bias=True)

    def forward(self, batch: torch.Tensor):
        x = batch['coords'] # b, c, n
        x = F.relu(self.bn1(self.conv1(x))).permute(0,2,1) # b, n, c
        x = self.tnn1(x)
        x = self.tnn2(x)
        x = self.tnn3(x)
        feature = self.tnn4(x).permute(0,2,1) # b, c, n
        # projection module
        x = F.adaptive_max_pool1d(feature, 1).squeeze() # b, c
        x = F.relu(self.bn2(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x, feature


class GTU(nn.Module):
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
        self.o = nn.Linear(d1, d1, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep = Tno(h=num_heads, dim=self.head_dim, rpe_dim=rpe_embedding, causal=causal, 
                        use_decay=use_decay, use_multi_decay=use_multi_decay, residual=residual,
                        act=rpe_act, par_type=par_type, gamma=gamma, bias=bias, act_type=act_type,
                        layers=rpe_layers, norm_type=norm_type)
        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)

    
    def forward(self, x):
        # b, n, d -> b, n, d
        # x: b, h, w, d
        num_heads = self.num_heads

        x = self.norm(x)

        u = self.act(self.u_proj(x)) # b, n, d1
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads) # b, h, n, d1
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> b n (h d)') # b, n, h*d1
        output = u * output
        output = self.o(output) # b, n, d1
                     
        return output