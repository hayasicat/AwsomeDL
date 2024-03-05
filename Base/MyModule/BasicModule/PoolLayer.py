# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 10:15
# @Author  : ljq
# @desc    : 
# @File    : PoolLayer.py
import torch
from torch import nn


class ChannelPool(nn.Module):
    def forward(self, x):
        x = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        return x
