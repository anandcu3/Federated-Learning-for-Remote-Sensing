from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class LENET(nn.Module):
    def __init__(self, n_classes):
        super(LENET, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5))
        self.linear1 = nn.Linear(64 * 24 * 24, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.

        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2, stride=2)
        #x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2, stride=2)
        x = x.view(-1, 64 * 24 * 24)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class RESNET34(nn.Module):
    def __init__(self, n_classes):
        super(RESNET34, self).__init__()
        self.model_ft = models.resnet34(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features

        self.model_ft.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        return self.model_ft(x)

class ALEXNET(nn.Module):
    def __init__(self, n_classes):
        super(ALEXNET, self).__init__()
        self.model_ft = models.alexnet(pretrained=False)
        num_ftrs = self.model_ft.classifier[4].out_features
        #self.model_ft.fc = nn.Linear(num_ftrs, n_classes)
        self.model_ft.classifier[6] = nn.Linear(num_ftrs, n_classes)
    def forward(self, x):
        return self.model_ft(x)