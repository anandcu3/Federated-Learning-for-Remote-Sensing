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

    
class VGG16(nn.Module):
    def __init__(self, numCls = 19):
        super().__init__()

        vgg = models.vgg16(pretrained=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            *vgg.features[1:]
        )
        self.classifier = nn.Sequential(
            nn.Linear(4608,4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls, bias=True)
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 


class VGG19(nn.Module):
    def __init__(self, numCls = 19):
        super().__init__()

        vgg = models.vgg19(pretrained=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            *vgg.features[1:]
        )
        self.classifier = nn.Sequential(
            nn.Linear(4608,4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls, bias=True)
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 




if __name__ == "__main__":
    
    inputs = torch.randn((4,10,120,120))
    # net = VGG16()
    net = VGG19()
    outputs = net(inputs)




