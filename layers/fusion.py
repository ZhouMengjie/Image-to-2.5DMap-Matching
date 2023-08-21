import torch
import torch.nn as nn
import timm
import sys
import os

class VanillaEncoder(nn.Module):
    def __init__(self, num_latents, pano_enc, map_enc):
        super(VanillaEncoder, self).__init__()

        # Pano
        # Attention Layer
        self.pano_norm1 = pano_enc.norm1
        self.pano_attn = pano_enc.attn
        # Feed Forward Layers
        self.pano_norm2 = pano_enc.norm2
        self.pano_mlp = pano_enc.mlp

        # Map
        # Attention Layer
        self.map_norm1 = map_enc.norm1
        self.map_attn = map_enc.attn
        # Feed Forward Layers
        self.map_norm2 = map_enc.norm2
        self.map_mlp = map_enc.mlp

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1,num_latents,768).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))


    def attention(self, q, k, v): # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x
    
    # Latent Fusion
    def fusion(self, pano_tokens, map_tokens):
        # shapes
        BS = pano_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((pano_tokens,map_tokens),dim=1)
        # cross attention (AV -->> latents)
        fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_)
        # cross attention (latents -->> AV)
        pano_tokens = pano_tokens + self.scale_a * self.attention(q=pano_tokens, k=fused_latents, v=fused_latents)
        map_tokens = map_tokens + self.scale_v * self.attention(q=map_tokens, k=fused_latents, v=fused_latents)
        return pano_tokens, map_tokens
    
    def forward(self, x, y):

        # Bottleneck Fusion
        x, y = self.fusion(x, y)

        # # Attn skip connections
        x = x + self.pano_attn(self.pano_norm1(x)) # qkv: linear(768, 2304//3), 12 heads
        y = y + self.map_attn(self.map_norm1(y))

        # FFN + skip conections
        x = x + self.pano_mlp(self.pano_norm2(x))
        y = y + self.map_mlp(self.map_norm2(y))
        return x, y


class Fusion(nn.Module):
    def __init__(self, in_channels=512, num_latents=4, depth=4):
        super(Fusion, self).__init__()
        self.pano = timm.create_model('vit_base_patch16_224_in21k', pretrained=True) # for pano
        self.map = timm.create_model('vit_base_patch16_224_in21k', pretrained=True) # for map

        """
        discard unnecessary layers and save parameters
        """
        self.pano.patch_embed.proj = nn.Identity()
        self.map.patch_embed.proj = nn.Identity()
        self.pano.pre_logits = nn.Identity()
        self.map.pre_logits = nn.Identity()
        self.pano.head = nn.Identity()
        self.map.head = nn.Identity()

        self.pano.pos_embed = nn.Parameter(torch.randn(1, 7*14, 768) * .02)
        self.map.pos_embed = nn.Parameter(torch.randn(1, 7*7, 768) * .02)
        # self.map.pos_embed = nn.Parameter(torch.randn(1, 1024, 768) * .02)

        """
        Initialize conv projection, cls token, pos embed and encoders for audio and visual modality
        """
        # conv projection
        self.pano_conv = nn.Conv2d(in_channels, 768, kernel_size=1)

        self.map_conv = nn.Conv2d(in_channels, 768, kernel_size=1) 
        # self.map_conv = nn.Conv1d(in_channels, 768, kernel_size=1)        

        # cls token and pos embedding
        self.pano_pos_embed = self.pano.pos_embed
        self.map_pos_embed = self.map.pos_embed

        self.pano_cls_token = self.pano.cls_token
        self.map_cls_token = self.map.cls_token

        """
        Initialize Encoder, Final Norm and Classifier
        """
        encoder_layers = []
        for i in range(depth):
            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoder(num_latents=num_latents, pano_enc=self.pano.blocks[i], map_enc=self.map.blocks[i]))
            
        self.fusion_blocks = nn.Sequential(*encoder_layers)

        # final norm
        self.pano_post_norm = self.pano.norm
        self.map_post_norm = self.map.norm


        """
        Forward pass for Spectrogram and RGB Images
        """
    def forward_pano_features(self, x): # shape = (bs, c, h, w)
        x = self.pano_conv(x) # shape = (bs, 768, 28, 56)
        B, dim, h, w = x.shape 
        x = torch.reshape(x,(B, dim, h*w)) # shape = (bs, 768, 1568)
        x = x.permute(0,2,1) # shape = (bs, 1568, 768)
        x = torch.cat((self.pano_cls_token.expand(B, -1, -1),x), dim=1) # shape = (bs, 1+1568, 768)
        # interpolate pos embedding and add
        x = x + nn.functional.interpolate(self.pano_pos_embed.permute(0,2,1), x.shape[1], mode='linear').permute(0,2,1)
        return x
        
    def forward_map_features(self, x):
        x = self.map_conv(x) # shape = (bs, 768, 28, 28) or (bs, 768, 1024)
        B, dim, h, w = x.shape 
        x = torch.reshape(x,(B, dim, h*w)) # shape = (bs, 768, 784)
        x = x.permute(0,2,1) # shape = (bs, 784, 768)
        x = torch.cat((self.map_cls_token.expand(B, -1, -1),x), dim=1) # shape = (bs, 1+784, 768)
        # interpolate pos embedding and add
        x = x + nn.functional.interpolate(self.map_pos_embed.permute(0,2,1), x.shape[1], mode='linear').permute(0,2,1)
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
    savedStdout = sys.stdout
    print_log = open(os.path.join('arun_log','fusion_arch.txt'),'w')
    sys.stdout = print_log
    n_params = 0
    for name, param in network.named_parameters():
        n_params += param.nelement()
        print('%14s: %s' % (name, param.nelement()))
    print('total parameters: {}'.format(n_params))
    # pano_data = torch.rand(4,512,28,56).cuda()
    # map_data = torch.rand(4,512,28,28).cuda()
    # pano_feat, map_feat = network(pano_data, map_data)
