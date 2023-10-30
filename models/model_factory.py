""" To use point cloud models with projection module, refer to pointnet, dgcnn... 
    To use point cloud models without projection module, refer to pointnet_v2, dgcnn_v2...
    To achieve pixel-to-point or point-to-pixel fusion, use point_fuse, dgcnn_fuse and tile_safa_fuse
"""
from models.multimodal import Multimodal
from config.utils import Params
from models.pointnet_v2 import PointNet
from models.pointnetssg_v2 import PointNetSSG
from models.dgcnn_fuse import DGCNN
from models.pt2_v2 import PT2
from models.pct_v2 import PCT
from models.tile import Tile
from models.pano import Pano
from models.transgeo import TransGeo
from models.pano_safa import Pano_SAFA
from models.tile_safa_fuse import Tile_SAFA
from models.transhybrid import TransHybrid

def model_factory(params: Params):
    if params.use_feat:
        in_channels = params.feat_size + 3
    else:
        in_channels = 3
    tile_fe, cloud_fe = None, None
    tile_fe_size, cloud_fe_size = 0, 0

    # select tile feature extractor
    if params.model_params.model2d_tile == 'resnet':
        tile_fe_size = params.feat_dim
        tile_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
    elif params.model_params.model2d_tile == 'resnet_safa':
        tile_fe_size = params.feat_dim
        tile_fe = Tile_SAFA(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
    elif params.model_params.model2d_tile == 'deit':
        tile_fe_size = params.feat_dim
        tile_fe = TransGeo(out_channels=tile_fe_size, img_size=params.model_params.tile_size, data_form='tile')
    elif params.model_params.model2d_tile == 'vit':
        tile_fe_size = params.feat_dim
        tile_fe = TransHybrid(out_channels=tile_fe_size, img_size=params.model_params.tile_size, data_form='tile')
    else:
        print('Model2D_Tile not implemented')
    
    # select cloud feature extractor
    if params.model_params.model3d == 'pointnet':
        cloud_fe_size = params.feat_dim
        cloud_fe = PointNet(out_channel=cloud_fe_size, in_channel=in_channels)
    elif params.model_params.model3d == 'pointnetssg':
        cloud_fe_size = params.feat_dim
        cloud_fe = PointNetSSG(out_channel=cloud_fe_size, normal_channel=params.use_feat, npoint=params.npoints, nneighbor=params.nneighbor)       
    elif params.model_params.model3d == 'dgcnn':
        cloud_fe_size = params.feat_dim
        cloud_fe = DGCNN(out_channel=cloud_fe_size, in_channel=in_channels, nneighbor=params.nneighbor)
    elif params.model_params.model3d == 'pt2':
        cloud_fe_size = params.feat_dim
        cloud_fe = PT2(out_channel=cloud_fe_size, in_channel=in_channels, npoint=params.npoints, nneighbor=params.nneighbor)      
    elif params.model_params.model3d == 'pct':
        cloud_fe_size = params.feat_dim
        cloud_fe = PCT(out_channel=cloud_fe_size, in_channel=in_channels, npoint=params.npoints, nneighbor=params.nneighbor)
    else:
        print('Model3D not implemented')        
    
    image_fe_size = params.feat_dim        
    if params.model_params.model2d_pano== 'resnet':
        image_fe = Pano(out_channels=image_fe_size, img_size=params.model_params.img_size)
    elif params.model_params.model2d_pano == 'resnet_safa':
        image_fe = Pano_SAFA(out_channels=image_fe_size, img_size=params.model_params.img_size)
    elif params.model_params.model2d_pano == 'deit':
        image_fe = TransGeo(out_channels=image_fe_size, img_size=params.model_params.img_size, data_form='pano')
    elif params.model_params.model2d_pano == 'vit':
        image_fe = TransHybrid(out_channels=image_fe_size, img_size=params.model_params.img_size, data_form='pano')
    else:
        print('Model2D_Pano not implemented')        

    model = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method=params.fuse, final_block=params.fc)
    return model
