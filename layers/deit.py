# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.pyplot as plt


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2

@register_model
def deit_tiny_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=192, num_classes=num_classes, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model_file = 'deit_tiny_distilled_patch16_224-b40b3cf7.pth'
        if not os.access(model_file, os.W_OK):
            weight_url = 'https://dl.fbaipublicfiles.com/deit/' + model_file
            os.system('wget ' + weight_url)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

        # resize the positional embedding
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])  
    return model

@register_model
def deit_small_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
        #     map_location="cpu", check_hash=True
        # )
        model_file = 'deit_small_distilled_patch16_224-649709d9.pth'
        if not os.access(model_file, os.W_OK):
            weight_url = 'https://dl.fbaipublicfiles.com/deit/' + model_file
            os.system('wget ' + weight_url)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

        # resize the positional embedding
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
    return model

