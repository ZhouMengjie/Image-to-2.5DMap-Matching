import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import hiddenlayer as h
from timm.models.layers import trunc_normal_
from torch.utils.tensorboard import SummaryWriter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(1, max_len, d_model)
        # pe[0, :, 0::2] = torch.sin(position * div_term)
        # pe[0, :, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)

        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x = x + self.pe[:x.size(1)]
        x = x + self.pe
        return self.dropout(x)

class TransMixer(nn.Module):
    def __init__(self, transDimension, hidden_dim=512, nHead=8, numLayers=1, max_length=5):
        '''
        Transformer for mixing street view features
        transDimension: transformer embedded dimension
        nHead: number of heads
        numLayers: number of encoded layers
        Return => features of the same shape as input
        '''
        super(TransMixer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(d_model=hidden_dim,\
            nhead=nHead, batch_first=True, dropout=0.1, norm_first = True, activation='relu')
        
        self.Transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

        self.embedding = nn.Linear(transDimension, hidden_dim)
        init.kaiming_normal_(self.embedding.weight)
        if self.embedding.bias is not None:
            init.constant_(self.embedding.bias, 0)

        self.positionalEncoding = PositionalEncoding(d_model=hidden_dim, max_len=max_length)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.positionalEncoding(x)
        x = self.Transformer(x)
        x = x.mean(dim=1)
        return x

class TransMixer_v2(nn.Module):
    def __init__(self, transDimension, hidden_dim=512, nHead=8, numLayers=1, max_length=5):
        '''
        Transformer for mixing street view features
        transDimension: transformer embedded dimension
        nHead: number of heads
        numLayers: number of encoded layers
        Return => features of the same shape as input
        '''
        super(TransMixer_v2, self).__init__()
        encoderLayer1 = nn.TransformerEncoderLayer(d_model=hidden_dim,\
            nhead=nHead, batch_first=True, dropout=0.1, norm_first = True, activation='relu')       
        self.Transformer1 = nn.TransformerEncoder(encoderLayer1, num_layers=numLayers)
        self.embedding1 = nn.Linear(transDimension, hidden_dim)
        init.kaiming_normal_(self.embedding1.weight)
        if self.embedding1.bias is not None:
            init.constant_(self.embedding1.bias, 0)
        self.positionalEncoding1 = PositionalEncoding(d_model=hidden_dim, max_len=max_length)

        encoderLayer2 = nn.TransformerEncoderLayer(d_model=hidden_dim,\
            nhead=nHead, batch_first=True, dropout=0.1, norm_first = True, activation='relu')       
        self.Transformer2 = nn.TransformerEncoder(encoderLayer2, num_layers=numLayers)
        self.embedding2 = nn.Linear(transDimension, hidden_dim)
        init.kaiming_normal_(self.embedding2.weight)
        if self.embedding2.bias is not None:
            init.constant_(self.embedding2.bias, 0)
        self.positionalEncoding2 = PositionalEncoding(d_model=hidden_dim, max_len=max_length)
    
    def forward(self, x, y):
        x = self.embedding1(x)
        x = self.positionalEncoding1(x)
        x = self.Transformer1(x)
        x = x.mean(dim=1)

        y = self.embedding2(y)
        y = self.positionalEncoding2(y)
        y = self.Transformer2(y)
        y = y.mean(dim=1)
        return x, y


class Smooth(nn.Module):
    def __init__(self):
        super(Smooth, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,None))
    
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class SeqNet(nn.Module):
    def __init__(self, inDims, outDims=512, seqL=5, w=3):
        super(SeqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(outDims, outDims, kernel_size=self.w)
        self.embedding = nn.Linear(inDims, outDims)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):    
        x = self.embedding(x)
        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        x = self.conv(x)
        x = torch.mean(x,-1)
        return x


class SeqNet_v2(nn.Module):
    def __init__(self, inDims, outDims=512, seqL=5, w=3):
        super(SeqNet_v2, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.embedding1 = nn.Linear(inDims, outDims)
        self.embedding2 = nn.Linear(inDims, outDims)
        self.conv1 = nn.Conv1d(outDims, outDims, kernel_size=self.w)
        self.conv2 = nn.Conv1d(outDims, outDims, kernel_size=self.w)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, y):  
        x = self.embedding1(x)   
        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        x = self.conv1(x)
        x = torch.mean(x,-1)
        
        y = self.embedding2(y)
        y = y.permute(0,2,1) # from [B,T,C] to [B,C,T]
        y = self.conv2(y)
        y = torch.mean(y,-1)
        return x, y


class Delta(nn.Module):
    def __init__(self, inDims, seqL=5):
        super(Delta, self).__init__()
        self.inDims = inDims
        self.weight = (np.ones(seqL,np.float32))/(seqL/2.0)
        self.weight[:seqL//2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(self.weight),requires_grad=False)

    def forward(self, x):
        # make desc dim as C
        x = x.permute(0,2,1) # makes [B,T,C] as [B,C,T]
        delta = torch.matmul(x,self.weight)
        return delta

class Baseline(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x 

  
if __name__ == "__main__":
    model = TransMixer_v2(4096,512).to('cuda')
    # model = SeqNet(4096).to('cuda')
    # model = Delta(4096).to('cuda')
    print(model)
    x = torch.rand((16, 5, 4096)).to('cuda')
    # vis_graph = h.build_graph(model, x)
    # vis_graph.theme = h.graph.THEMES["blue"].copy() 
    # vis_graph.save('model.png') 
    # writer = SummaryWriter("model_logs/")
    # writer.add_graph(model, x)
    # writer.close()
    feat = model(x)
    print(feat.shape)


