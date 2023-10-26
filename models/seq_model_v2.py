import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dgcnn_fuse import DGCNN
from models.pano_safa import Pano_SAFA
from models.tile_safa_fuse import Tile_SAFA
from models.multimodal import Multimodal
from models.seq_model import TransMixer
from models.seq_model import TransMixer_v2
from models.seq_model import SeqNet
from models.seq_model import SeqNet_v2


class SeqModel(nn.Module):
    def __init__(self, params):
        super(SeqModel, self).__init__()
        if params.map_type == 'single':
            tile_fe_size = params.encoder_dim
            tile_fe = Tile_SAFA(out_channels=tile_fe_size, tile_size=params.tile_size, use_polar=params.use_polar)
            image_fe_size = params.encoder_dim    
            image_fe = Pano_SAFA(out_channels=image_fe_size, img_size=params.image_size)
            cloud_fe = None
            cloud_fe_size = 0
            self.encoder = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method='concat')
        elif params.map_type == 'multi':
            tile_fe_size, cloud_fe_size, image_fe_size = params.encoder_dim, params.encoder_dim, params.encoder_dim
            tile_fe = Tile_SAFA(out_channels=tile_fe_size, tile_size=params.tile_size, use_polar=params.use_polar)
            cloud_fe = DGCNN(out_channel=cloud_fe_size, in_channel=3, nneighbor=20)
            image_fe = Pano_SAFA(out_channels=image_fe_size, img_size=params.image_size)
            self.encoder = Multimodal(cloud_fe, cloud_fe_size, image_fe, image_fe_size, tile_fe, tile_fe_size, output_dim=image_fe_size, fuse_method='2to3')
        else:
            raise NotImplementedError('Unsupported Map Type: {}'.format(params.map_type))

        # Initialize encoder parameters
        if params.pretrained is not None:
            checkpoint = torch.load(params.pretrained, map_location=params.device)  
            if 'model' in checkpoint:
                ckp = checkpoint['model']
            else:
                ckp = checkpoint
            state_dict = {}
            for k, v in ckp.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                state_dict[new_k] = v
            self.encoder.load_state_dict(state_dict, strict=True)
            print('load pretrained {} encoder!'.format(params.pretrained)) 

        if params.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False  # To freeze the parameters

        if params.model_type == 'transmixer' and params.share:
            self.seqmodel = TransMixer(params.feat_dim, nHead=params.num_heads, numLayers=params.num_layers, max_length=params.seq_len)
        elif params.model_type == 'transmixer' and not params.share:
            self.seqmodel = TransMixer_v2(params.feat_dim, nHead=params.num_heads, numLayers=params.num_layers, max_length=params.seq_len)
        elif params.model_type == 'seqnet' and params.share:
            self.seqmodel = SeqNet(params.feat_dim, seqL=params.seq_len)
        elif params.model_type == 'seqnet' and not params.share:
            self.seqmodel = SeqNet_v2(params.feat_dim, seqL=params.seq_len)

        self.seq_len = params.seq_len
        self.share = params.share

    def forward(self, batch):
        x = self.encoder(batch)
        image_embedding = x['image_embedding']
        map_embedding = x['embedding']
        
        image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=1)  # Normalize embeddings
        map_embedding = torch.nn.functional.normalize(map_embedding, p=2, dim=1)
        
        image_embedding = image_embedding.view(-1, self.seq_len, image_embedding.shape[1])
        map_embedding = map_embedding.view(-1, self.seq_len, map_embedding.shape[1])
        if self.share:
            pano_feat = self.seqmodel(image_embedding)
            map_feat = self.seqmodel(map_embedding)
        else:
            pano_feat, map_feat = self.seqmodel(image_embedding, map_embedding)
        return pano_feat, map_feat



        


        
    


        

