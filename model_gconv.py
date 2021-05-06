import torch
import torch.nn as nn
import math
from gconv import ConvZ2P4, ConvP4, MaxSpatialPoolP4, AvgRootPoolP4
# from groupy.gconv.pytorch_gconv import P4ConvP4, P4ConvZ2

class ConvDouble(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvDouble, self).__init__()
        
        # self.conv1 = P4ConvP4(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv1 = ConvP4(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        # self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm1 = nn.GroupNorm(8, out_channel)
        # self.conv2 = P4ConvP4(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv2 = ConvP4(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        # self.norm2 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.GroupNorm(8, out_channel)
        self.relu = nn.ReLU(inplace=True)
        # self.res = nn.Identity()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        # residual = self.res(x)
        x = self.relu(self.norm2(self.conv2(x)))
        return x# + residual


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()

        self.convdouble = ConvDouble(in_channel=in_channel, out_channel=out_channel)
        self.pool = MaxSpatialPoolP4(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.convdouble(x)
        x = self.pool(x1)
        return x, x1


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        
        # self.upconv = P4ConvP4(in_channels=in_channel, out_channels=in_channel//2, kernel_size=2, stride=2, transpose=True)
        self.upconv = ConvP4(in_channels=in_channel, out_channels=in_channel//2, kernel_size=2, stride=2, transpose=True)
        self.convdouble = ConvDouble(in_channel=in_channel, out_channel=out_channel)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diff_x = x2.size(-2) - x1.size(-2)
        diff_y = x2.size(-1) - x1.size(-1)
        if x1.size() != x2.size():
            pad = F.pad(x1, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
            ])
        else:
            pad = x1
        x = torch.cat((x2, pad), 1)
        x = self.convdouble(x)
        return x


class UNetGConv(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetGConv, self).__init__()

        # self.conv_first = P4ConvZ2(n_channels, n_channels, kernel_size=7, padding=3)
        self.conv_first = ConvZ2P4(n_channels, n_channels, kernel_size=7, padding=3)

        self.down1 = DownSample(n_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = DownSample(64, 128)
        self.down4 = DownSample(128, 256)

        self.convdouble = ConvDouble(256, 512)
        self.drop = nn.Dropout()

        self.up1 = UpSample(512, 256)
        self.up2 = UpSample(256, 128)
        self.up3 = UpSample(128, 64)
        self.up4 = UpSample(64, 32)

        # self.conv_last = P4ConvP4(32, n_classes, kernel_size=3, padding=1)
        self.conv_last = ConvP4(32, n_classes, kernel_size=3, padding=1)
        self.normalize = nn.Sigmoid() if n_classes == 1 else nn.Softmax(1)
        self.convert_z2 = AvgRootPoolP4()
    
    def forward(self, x):
        x = self.conv_first(x)

        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)

        x = self.convdouble(x)
        x = self.drop(x)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.conv_last(x)
        x = self.normalize(x)
        x = self.convert_z2(x)
        return x

    def total_params(self):
        total = sum(p.numel() for p in self.parameters())
        return format(total, ',')

    def total_trainable_params(self):
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return format(total_trainable, ',')