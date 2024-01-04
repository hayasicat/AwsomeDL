# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 9:39
# @Author  : ljq
# @desc    : 跟着论文敲的efficientV2实在是起不来，看看效果
# @File    : EfficientM.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._utils import Conv3x3, PointWiseConv, DepthWiseConv, DPWConv, MBConv


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            PointWiseConv(in_channels, in_channels * t),
            DepthWiseConv(in_channels * t, in_channels * t, stride),
            PointWiseConv(in_channels * t, out_channels, act=nn.Identity),
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class EMBConv(MBConv):
    def __init__(self, input_channel, output_channel, stride=1, expand_ratio=4):
        super().__init__(input_channel, output_channel, stride=stride, expand_ratio=expand_ratio)


class EfficientModify(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()
        # TODO: 这边的起手stride应该为2, 对齐32的缩放率看看效果,小图片起手应该是stride 1
        self.pre = Conv3x3(3, 32, stride=1)
        # 这边得downsample次数比较少
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6, EMBConv)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6, EMBConv)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6, EMBConv)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6, EMBConv)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6, EMBConv)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = PointWiseConv(320, 1280)
        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, Block):

        layers = []
        layers.append(Block(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(Block(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

    def init_weights(self):
        # 初始化卷积和BatchNorm
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # layer.bias.data.fill_(0.001)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
