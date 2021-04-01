import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3,3), padding='same', activation=True):
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channel):
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.relu = nn.LeakyReLU(0.1)
        self.convblock1 = ConvBlock(channel, channel)
        self.convblock2 = ConvBlock(channel, channel, activation=False)

    def forward(self, x):
        x = self.norm1(x)
        res = self.relu(x)
        res = self.norm2(res)
        res = self.convblock1(res)
        res = self.convblock2(res)
        return res + x

class UEfficientNet(nn.Module):
    def __init__(self):
        pass
    pass
