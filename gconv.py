import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvZ2P4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=1):
        super().__init__()
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def _rotated(self, w):
        ws = [torch.rot90(w, k, (2, 3)) for k in range(4)]
        return torch.cat(ws, 1).view(-1, w.size(1), w.size(2), w.size(3))

    def forward(self, x):
        w = self._rotated(self.weight)
        y = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class ConvP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=1, transpose=False):
        super().__init__()
        w = torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.transpose = transpose

    def _grot90(self, x, k):
        return torch.rot90(x.roll(k, 2), k, (3, 4))

    def _rotated(self, w):
        ws = [self._grot90(w, k).view(w.size(0), -1, w.size(3), w.size(4)) for k in range(4)]
        return torch.cat(ws, 1).view(4 * w.size(0), 4 * w.size(1), w.size(3), w.size(4))

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        w = self._rotated(self.weight)
        if self.transpose:
            y = F.conv_transpose2d(s, w, stride=self.stride, padding=self.padding)
        else:
            y = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        y = y.view(y.size(0), -1, 4, y.size(2), y.size(3))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class ConvP4Z2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=1):
        super().__init__()
        w = torch.empty(out_channels, 4*in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        w = self.weight
        y = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class MaxRotationPoolP4(nn.Module):
    def forward(self, x):
        return x.max(2).values


class MaxSpatialPoolP4(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.inner = nn.MaxPool2d(kernel_size, stride, padding)
    
    def forward(self, x):
        y = x.view(x.size(0), -1, x.size(3), x.size(4))
        y = self.inner(y)
        y = y.view(x.size(0), -1, 4, y.size(2), y.size(3))
        return y


class AvgRootPoolP4(nn.Module):
    def forward(self, x):
        return x.mean(2)