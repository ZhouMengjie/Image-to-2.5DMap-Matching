import torch.nn as nn
from layers.deit import deit_small_distilled_patch16_224, deit_tiny_distilled_patch16_224

class TransGeo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, out_channels, img_size, data_form):
        """
        dim: feature dimension (default: 512)
        """
        super(TransGeo, self).__init__()
        self.dim = out_channels
        self.data_form = data_form

        if data_form == 'pano':
            self.size = [img_size, img_size*2]
            base_model = deit_small_distilled_patch16_224
        elif data_form == 'tile':
            self.size = [img_size, img_size]
            base_model = deit_small_distilled_patch16_224

        # create the encoders
        # num_classes is the output fc dimension
        self.net = base_model(img_size=self.size, num_classes=out_channels)

    def forward(self, batch):
        if self.data_form == 'pano':
            x = batch['images']
        elif self.data_form == 'tile':
            x = batch['tiles']
        return self.net(x), None
