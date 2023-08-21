import torch.nn as nn
import torch

class SAFA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask


