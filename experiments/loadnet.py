import os
import sys
sys.path.append(os.getcwd())
import torch.nn as nn
from torchsummary import summary
from config.utils import get_datetime
from models.pano import Pano
from models.tile import Tile


if __name__ == '__main__':
    savedStdout = sys.stdout
    s = get_datetime()
    print_log = open(os.path.join('arun_log',s+'.txt'),'w')
    sys.stdout = print_log
    # model = Tile(128,128)
    model = nn.Conv1d(8192, 4096, 1, bias=True)
    # summary(model,input_size=(3,224,224))
    param_dict = vars(model)
    for e in param_dict:
        print('{}: {}'.format(e, param_dict[e]))

