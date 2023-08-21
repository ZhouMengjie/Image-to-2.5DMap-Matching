# Author: Jacek Komorowski
# Warsaw University of Technology
from models.multimodal import Multimodal
from config.utils import MinkLocParams
from models.pointnet import PointNet
from models.pointnetssg import PointNetSSG
from models.dgcnn import DGCNN
from models.pt2 import PT2
from models.pct import PCT
from models.tile import Tile
from models.pano import Pano
from models.pt import PT
from models.transgeo import TransGeo
from models.pano_safa import Pano_SAFA
from models.tile_safa import Tile_SAFA
from models.transmap import TransMap

def model_factory(params: MinkLocParams):
    if params.use_feat:
        in_channels = params.feat_size + 3
    else:
        in_channels = 3
    tile_fe, cloud_fe = None, None
    tile_fe_size, cloud_fe_size = 0, 0
    if params.model_params.model == 'MinkLocMultimodal':
        if params.model_params.model3d == 'none':
            tile_fe_size = params.feat_dim
            # tile_fe = Tile_SAFA(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            tile_fe = TransMap(out_channels=tile_fe_size, img_size=params.model_params.tile_size, data_form='tile')
        elif params.model_params.model3d == 'cloud_pano':
            cloud_fe_size = params.feat_dim
            cloud_fe = PointNet(out_channel=cloud_fe_size, in_channel=in_channels)
        elif params.model_params.model3d == 'cloud_tile':
            cloud_fe_size = params.feat_dim
            cloud_fe = PointNet(out_channel=cloud_fe_size, in_channel=in_channels)
        elif params.model_params.model3d == 'pointnet':
            tile_fe_size, cloud_fe_size = params.feat_dim, params.feat_dim
            tile_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            cloud_fe = PointNet(out_channel=cloud_fe_size, in_channel=in_channels)
        elif params.model_params.model3d == 'pointnetssg':
            tile_fe_size, cloud_fe_size = params.feat_dim, params.feat_dim
            tile_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            cloud_fe = PointNetSSG(out_channel=cloud_fe_size, normal_channel=params.use_feat, npoint=params.npoints, nneighbor=params.nneighbor)       
        elif params.model_params.model3d == 'dgcnn':
            tile_fe_size, cloud_fe_size = params.feat_dim, params.feat_dim
            tile_fe = Tile_SAFA(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            cloud_fe = DGCNN(out_channel=cloud_fe_size, in_channel=in_channels, nneighbor=params.nneighbor)
        elif params.model_params.model3d == 'pt2':
            tile_fe_size, cloud_fe_size = params.feat_dim, params.feat_dim
            tile_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            cloud_fe = PT2(out_channel=cloud_fe_size, in_channel=in_channels, npoint=params.npoints, nneighbor=params.nneighbor)
        elif params.model_params.model3d == 'pt':
            tile_fe_size, cloud_fe_size = params.feat_dim, params.feat_dim
            tile_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            cloud_fe = PT(out_channel=cloud_fe_size, in_channel=in_channels)       
        elif params.model_params.model3d == 'pct':
            tile_fe_size, cloud_fe_size = params.feat_dim, params.feat_dim
            tile_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
            cloud_fe = PCT(out_channel=cloud_fe_size, in_channel=in_channels, npoint=params.npoints, nneighbor=params.nneighbor)
        else:
            raise NotImplementedError('Model3D not implemented: {}'.format(params.model_params.model3d))        
        image_fe_size = params.feat_dim
        
        if params.model_params.model3d == 'cloud_tile':
            image_fe = Tile(out_channels=tile_fe_size, tile_size=params.model_params.tile_size, use_polar=params.use_polar)
        else:
            # image_fe = Pano_SAFA(out_channels=image_fe_size, img_size=params.model_params.img_size)
            image_fe = TransMap(out_channels=image_fe_size, img_size=params.model_params.img_size, data_form='pano')
        # model = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method=params.fuse, final_block='fc', regularizer=params.use_regu)
        model = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method=params.fuse, regularizer=params.use_regu)
    return model
