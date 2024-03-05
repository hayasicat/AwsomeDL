# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 11:04
# @Author  : ljq
# @desc    : 
# @File    : Seg.py
import torch
from torch import nn


class SegHead(nn.Module):
    def __init__(self, input_channel, cls, activation_type='sigmoid'):
        super().__init__()
        if activation_type == 'sigmoid':
            activation = nn.Sigmoid()
        elif cls == 'softmax':
            activation = nn.Softmax(dim=1)
        else:
            activation = nn.Sequential()

        # 加上一个3x3的卷积作为感受野
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, cls, kernel_size=3, stride=1),
            activation,
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)
