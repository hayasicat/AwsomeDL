# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 17:56
# @Author  : ljq
# @desc    : 
# @File    : Conv.py
from torch import nn


class ConvBnAct(nn.Module):
    group = 1

    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, act=nn.ReLU):
        super().__init__()
        # 修复Group没用的Bug
        self.conv = nn.Conv2d(input_channel, output_channel, (kernel_size, kernel_size), stride=stride, padding=padding,
                              groups=self.group)
        self.bn = nn.BatchNorm2d(output_channel)
        if act is nn.ReLU:
            self.act = act(inplace=True)
        else:
            self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

