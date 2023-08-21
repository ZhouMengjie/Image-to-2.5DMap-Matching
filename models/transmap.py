import torch.nn as nn
from layers.vvt import vvt_small

class TransMap(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, out_channels, img_size, data_form, drop=0.0, drop_path=0.3):
        """
        dim: feature dimension (default: 512)
        """
        super(TransMap, self).__init__()
        self.data_form = data_form
        if data_form == 'pano':
            self.size = [img_size, img_size*2]
        elif data_form == 'tile':
            self.size = [img_size, img_size]

        # create the encoders
        # num_classes is the output fc dimension
        base_model = vvt_small
        self.net = base_model(pretrained=True, num_classes=out_channels, drop_rate=drop,
                             drop_path_rate=drop_path)

    def forward(self, batch):
        if self.data_form == 'pano':
            x = batch['images']
        elif self.data_form == 'tile':
            x = batch['tiles']
        return self.net(x), None
