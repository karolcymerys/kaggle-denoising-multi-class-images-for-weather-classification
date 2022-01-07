import torch
from torch import nn
import torch.nn.functional as F


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(256, 64, kernel_size=(2, 2), stride=(2, 2))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(64, 3, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(F.relu(self.conv2(x1)))

        x3 = F.relu(self.conv3(x2))
        x4 = self.pool(F.relu(self.conv4(x3)))

        x5 = F.relu(self.t_conv1(x4) + x2)
        x6 = self.conv5(self.upsample(x5))

        return torch.sigmoid(x6)
