import os
import sys
import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer, Attention

class VanillaEncoder(nn.Module):
    def __init__(self, num_latents, hidden_size):
        super(VanillaEncoder, self).__init__()
        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1,num_latents,hidden_size).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def attention(self, q, k, v): # requires q,k,v to have same dim
        B, N, C = q.shape
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    def forward(self, pano_tokens, map_tokens):
        # shapes
        BS = pano_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((pano_tokens,map_tokens),dim=1)
        # cross attention (PM -->> latents)
        fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_)
        # cross attention (latents -->> PM)
        pano_tokens = pano_tokens + self.scale_a * self.attention(q=pano_tokens, k=fused_latents, v=fused_latents)
        map_tokens = map_tokens + self.scale_v * self.attention(q=map_tokens, k=fused_latents, v=fused_latents)
        return pano_tokens, map_tokens

class Fusion(nn.Module):
    def __init__(self, in_channels=512, num_latents=4, depth=4):
        super(Fusion, self).__init__()
        # initialize cls token and pos embedding
        self.pano_pos_embed = nn.Parameter(torch.randn(1, 28*56, in_channels))
        # self.map_pos_embed = nn.Parameter(torch.randn(1, 7*7, in_channels))
        self.map_pos_embed = nn.Parameter(torch.randn(1, 1024, in_channels))

        self.pano_cls_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.map_cls_token = nn.Parameter(torch.zeros(1, 1, in_channels))

        # self.dropout = Dropout(0.1)

        # initialize Encoder and Final Norm
        encoder_layers = []
        for i in range(depth):
            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoder(num_latents=num_latents, hidden_size=in_channels))            
        self.fusion_blocks = nn.Sequential(*encoder_layers)

        # final norm
        self.pano_post_norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.map_post_norm = nn.LayerNorm(in_channels, eps=1e-6)

        # initialize weights
        trunc_normal_(self.pano_pos_embed, std=.02)
        trunc_normal_(self.map_pos_embed, std=.02)
        nn.init.normal_(self.pano_cls_token, std=1e-6)
        nn.init.normal_(self.map_cls_token, std=1e-6)
        self.init_weights(self.fusion_blocks)
   
    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()

    def forward_pano_features(self, x): # shape = (bs, c, h, w)
        B, dim, h, w = x.shape 
        x = torch.reshape(x,(B, dim, h*w)) # shape = (bs, c, h*w)
        x = x.permute(0,2,1) # shape = (bs, h*w, c)
        x = torch.cat((self.pano_cls_token.expand(B, -1, -1),x), dim=1) # shape = (bs, 1+h*w, c)
        # interpolate pos embedding and add
        x = x + nn.functional.interpolate(self.pano_pos_embed.permute(0,2,1), x.shape[1], mode='linear').permute(0,2,1) # shape = (bs, 1+h*w, c) 
        # x = self.dropout(x)
        return x
        
    def forward_map_features(self, x):
        # B, dim, h, w = x.shape 
        # x = torch.reshape(x,(B, dim, h*w)) 
        B, dim, n = x.shape
        x = x.permute(0,2,1) 
        x = torch.cat((self.map_cls_token.expand(B, -1, -1),x), dim=1) 
        # interpolate pos embedding and add
        x = x + nn.functional.interpolate(self.map_pos_embed.permute(0,2,1), x.shape[1], mode='linear').permute(0,2,1)
        # x = self.dropout(x)
        return x

    def forward_encoder(self, x, y):     
        # encoder forward pass
        for blk in self.fusion_blocks:
            x, y = blk(x, y)

        x = self.pano_post_norm(x)
        y = self.map_post_norm(y)

        # return class token alone
        x = x[:, 0]
        y = y[:, 0]
        return x, y
        
    def forward(self, x, y):
        x = self.forward_pano_features(x)
        y = self.forward_map_features(y)
        x, y = self.forward_encoder(x, y)
        return x, y

if __name__ == '__main__':
    network = Fusion(in_channels=512).cuda()
    # savedStdout = sys.stdout
    # print_log = open(os.path.join('arun_log','fusion_arch.txt'),'w')
    # sys.stdout = print_log
    n_params = 0
    for name, param in network.named_parameters():
        n_params += param.nelement()
        print('%14s: %s' % (name, param.nelement()))
    print('total parameters: {}'.format(n_params))
    pano_data = torch.rand(4,512,7,14).cuda()
    map_data = torch.rand(4,512,7,7).cuda()
    pano_feat, map_feat = network(pano_data, map_data)
