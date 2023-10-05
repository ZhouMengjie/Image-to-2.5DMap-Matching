import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TransMixer(nn.Module):
    def __init__(self, transDimension, nHead=8, numLayers=6, max_length = 5):
        '''
        Transformer for mixing street view features
        transDimension: transformer embedded dimension
        nHead: number of heads
        numLayers: number of encoded layers
        Return => features of the same shape as input
        '''
        super(TransMixer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(d_model = transDimension,\
            nhead=nHead, batch_first=True, dropout=0.3, norm_first = True)

        self.Transformer = nn.TransformerEncoder(encoderLayer, \
            num_layers=numLayers, \
            norm=nn.LayerNorm(normalized_shape=transDimension, eps=1e-6))

        self.positionalEncoding = PositionalEncoding(d_model = transDimension, max_len=max_length)
    
    def forward(self, x):
        x = self.positionalEncoding(x)
        x = self.Transformer(x)
        x = x.mean(dim=2)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Smooth(nn.Module):
    def __init__(self):
        super(Smooth).__init__()
        self.smooth = nn.Sequential(nn.AdaptiveAvgPool2d((1,None)), Flatten())
    
    def forward(self, x):
        x = self.smooth(x)
        return x


class SeqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=3):
        super(SeqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):       
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]
        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        x = self.conv(x)
        x = torch.mean(x,-1)
        return x


class Delta(nn.Module):
    def __init__(self, inDims, seqL):
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

  
if __name__ == "__main__":
    model = TransMixer(4096).to('cuda')
    print(model)
    x = torch.rand((16, 5, 4096)).to('cuda')
    feat = model(x)
    print(sq_descriptor.shape)


