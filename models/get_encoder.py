import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dgcnn_fuse import DGCNN
from models.pano_safa import Pano_SAFA
from models.tile_safa_fuse import Tile_SAFA
from models.multimodal import Multimodal


def get_encoder(params):
    if params.map_type == 'single':
        tile_fe_size = params.feat_dim
        tile_fe = Tile_SAFA(out_channels=tile_fe_size, tile_size=params.tile_size, use_polar=params.use_polar)
        image_fe_size = params.feat_dim    
        image_fe = Pano_SAFA(out_channels=image_fe_size, img_size=params.image_size)
        cloud_fe = None
        cloud_fe_size = 0
        encoder = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method='2to3')
    elif params.map_type == 'multi':
        tile_fe_size, cloud_fe_size, image_fe_size = params.feat_dim, params.feat_dim, params.feat_dim
        tile_fe = Tile_SAFA(out_channels=params.feat_dim, tile_size=params.tile_size, use_polar=params.use_polar)
        cloud_fe = DGCNN(out_channel=params.feat_dim, in_channel=3, nneighbor=20)
        image_fe = Pano_SAFA(out_channels=params.feat_dim, img_size=params.image_size)
        encoder = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method='2to3')
    else:
        raise NotImplementedError('Unsupported Map Type: {}'.format(params.map_type))
    return encoder
    


        

