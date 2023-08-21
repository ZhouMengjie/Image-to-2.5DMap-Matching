import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, out_channel, in_channel, embedding_channel=2048):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embedding_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding_channel)
        # projection module
        self.linear1 = nn.Conv1d(embedding_channel, 1024, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Conv1d(1024, out_channel, kernel_size=1, bias=True)

    def forward(self, batch: torch.Tensor):
        x = batch['coords']
        xyz = batch['xyz']
        center = batch['center']
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        feature = F.relu(self.bn5(self.conv5(x)))
        x = feature
        # projection module
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x, feature, xyz, center