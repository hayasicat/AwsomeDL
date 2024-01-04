# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 9:44
# @Author  : ljq
# @desc    : 
# @File    : ConvLayer.py
import torch.nn as nn


class Conv3x3(nn.Module):
    kernel_size = (3, 3)
    group = 1

    def __init__(self, input_channel, output_channel, stride=1, padding=1, act=nn.ReLU):
        super().__init__()
        # 修复Group没用的Bug
        self.conv = nn.Conv2d(input_channel, output_channel, self.kernel_size, stride=stride, padding=padding,
                              groups=self.group)
        self.bn = nn.BatchNorm2d(output_channel)
        if act is nn.ReLU:
            self.act = act(inplace=True)
        else:
            self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PointWiseConv(Conv3x3):
    def __init__(self, input_channel, output_channel, stride=1, padding=1, act=nn.ReLU):
        self.kernel_size = (1, 1)
        super().__init__(input_channel, output_channel, stride, 0, act)


class DepthWiseConv(Conv3x3):
    def __init__(self, input_channel, output_channel, stride=1, padding=1, act=nn.ReLU):
        self.group = input_channel
        super().__init__(input_channel, output_channel, stride, padding, act)


class DPWConv(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, padding=1, act=nn.ReLU):
        super(DPWConv, self).__init__()
        model = [
            DepthWiseConv(input_channel, input_channel, stride, padding, act),
            PointWiseConv(input_channel, output_channel),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input_tensor):
        return self.model(input_tensor)


