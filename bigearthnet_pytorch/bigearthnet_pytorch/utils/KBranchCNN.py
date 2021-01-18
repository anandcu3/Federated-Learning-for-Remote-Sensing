import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)



class ConvBNRelu(nn.Module):
    def __init__(self, in_, out, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class KBranchCNN(nn.Module):
    def __init__(self, numClass=19, feature_maps=[32, 32, 64]):
        super().__init__()

        self.feature_maps = feature_maps
        self.numLayers = len(feature_maps)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

        self.branch10_convs = nn.Sequential(
            ConvBNRelu(4, self.feature_maps[0], kernel_size=5, padding=2, stride=1),
            self.maxpool,
            self.dropout,
            ConvBNRelu(self.feature_maps[0], self.feature_maps[1], kernel_size=5, padding=2, stride=1),
            self.maxpool,
            self.dropout,
            ConvBNRelu(self.feature_maps[1], self.feature_maps[2], kernel_size=3),
            self.dropout
        )


        self.branch20_convs = nn.Sequential(
            ConvBNRelu(6, self.feature_maps[0], kernel_size=3),
            self.maxpool,
            self.dropout,
            ConvBNRelu(self.feature_maps[0], self.feature_maps[1], kernel_size=3),
            self.dropout,
            ConvBNRelu(self.feature_maps[1], self.feature_maps[2], kernel_size=3),
            self.dropout

        )

        self.FC_10 = nn.Linear(57600, 128)
        self.FC_20 = nn.Linear(57600, 128)
        self.FC = nn.Linear(256, 128)

        self.classification = nn.Linear(128, numClass)

        self.initialize()

    def initialize(self):
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x10, x20):

        x10 = self.branch10_convs(x10)

        x20 = self.branch20_convs(x20)

        x10 = x10.view(x10.size(0), -1)
        x20 = x20.view(x10.size(0), -1)

        x10 = self.FC_10(x10)
        x20 = self.FC_20(x20)
        
        x10 = self.relu(x10)
        x20 = self.relu(x20)

        x10 = self.dropout(x10)
        x20 = self.dropout(x20)

        x_f = torch.cat((x10, x20), dim=1)
        x_f = self.FC(x_f)
        x_f = self.relu(x_f)
        x_f = self.dropout(x_f)

        logits = self.classification(x_f)

        return logits



if __name__ == "__main__":
    
    bands10 = torch.randn(10,4,120,120)
    bands20 = torch.randn(10,6,60,60)

    net = KBranchCNN()

    outputs = net(bands10, bands20)

    print(outputs.shape)
















