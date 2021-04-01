import torch
import torch.nn as nn
import math

class ConvDouble(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvDouble, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1)
        # self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm1 = nn.GroupNorm(8, out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3), padding=1)
        # self.norm2 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.GroupNorm(8, out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()

        self.convdouble = ConvDouble(in_channel=in_channel, out_channel=out_channel)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x1 = self.convdouble(x)
        x = self.pool(x1)
        return x, x1


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=(2, 2), stride=2)
        self.convdouble = ConvDouble(in_channel=in_channel, out_channel=out_channel)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diff_x = x2.size(2) - x1.size(2)
        diff_y = x2.size(3) - x2.size(2)
        if x1.size() != x2.size():
            pad = torch.zeros(x2.size())
            pad[:, diff_x//2:-(diff_x-diff_x//2), diff_y//2:-(diff_y-diff_y//2), :] = x1
        else:
            pad = x1
        x = torch.cat((x2, pad), 1)
        x = self.convdouble(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super(UNet, self).__init__()

        self.down1 = DownSample(n_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.convdouble = ConvDouble(512, 1024)
        self.drop = nn.Dropout()

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.convlast = nn.Conv2d(64, n_class, kernel_size=(1, 1))

        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # force float division, therefore use 2.0
                # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
                # https://arxiv.org/abs/1502.01852
                # a rectifying linear unit is zero for half of its input,
                # so you need to double the size of weight variance to keep the signals variance constant.
                # xavier would be: scalefactor * sqrt(2/ (inchannels + outchannels )
                std = math.sqrt(2.0/(m.kernel_size[0]*m.kernel_size[0]*m.in_channels))
                nn.init.normal_(m.weight, std=std)
                #nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
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
        x = self.convlast(x)
        return x