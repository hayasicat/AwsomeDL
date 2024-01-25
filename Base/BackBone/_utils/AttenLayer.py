# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 10:42
# @Author  : ljq
# @desc    : 
# @File    : AttenLayer.py
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, input_channel, ratio=0.25):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(input_channel, int(input_channel * ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(input_channel * ratio), input_channel, 1, bias=False),
            nn.Sigmoid()
        ])

    def forward(self, x):
        out = self.model(x)
        return out * x


class MultiHeadAtten(nn.Module):
    pass
