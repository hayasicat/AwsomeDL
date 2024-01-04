# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 10:42
# @Author  : ljq
# @desc    : 
# @File    : AbsLayer.py
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth

from .ConvLayer import Conv3x3, PointWiseConv, DPWConv, DepthWiseConv
from .AttenLayer import SELayer


class MBConv(nn.Module):
    def __init__(self, input_channel, output_channel, expand_ratio=4, stride=1, has_skip=True, dropout_ratio=0.0):
        super().__init__()
        inter_media_channel = input_channel * expand_ratio
        # 最小限制要8层
        se_inter_media_channel = max([inter_media_channel, 8])
        # 校验一下如果stride>1也是不会有has_skip
        self.has_skip = has_skip
        if stride > 1:
            self.has_skip = False
        model = [
            PointWiseConv(input_channel, inter_media_channel),
            DepthWiseConv(inter_media_channel, inter_media_channel, stride),
            # nn.Dropout(dropout_ratio),
            SELayer(inter_media_channel),
            PointWiseConv(inter_media_channel, output_channel, act=nn.Identity),
        ]
        self.model = nn.Sequential(*model)
        self.stochastic_depth = StochasticDepth(dropout_ratio, 'row') if self.has_skip else nn.Identity
        # drop不太靠谱有
        # self.drop = nn.Dropout(dropout_ratio) if self.has_skip else nn.Identity

        # 不太懂是不是有stride和初始的没有连接在一起导致过拟合了，强行使用shkip_connection连接降采样层还是会过拟合。

    def forward(self, input_tensor):
        x = input_tensor
        out = self.model(x)
        if self.has_skip:
            return input_tensor + self.stochastic_depth(out)
            # TODO： dropout还是得添加在input_tensor这边精度才更高一点
            # return self.drop(input_tensor) + out
        # TODO： 原始的模型是没有这个结构的，这个结构是不是决定于能否提高对于底层连接的重要
        return out


class FuseMBConv(nn.Module):
    """
    #TODO： 为什么最后一层卷积没有ACT  -> mobilenetV2
    """

    def __init__(self, input_channel, output_channel, expand_ratio=4, stride=1, has_skip=True, dropout_ratio=0.1):
        super().__init__()
        inter_media_channel = input_channel * expand_ratio
        self.has_skip = has_skip
        model = [
            Conv3x3(input_channel, inter_media_channel, stride),
            # nn.Dropout(dropout_ratio),
            SELayer(inter_media_channel),
            PointWiseConv(inter_media_channel, output_channel, act=nn.Identity)
        ]
        self.model = nn.Sequential(*model)
        self.stochastic_depth = StochasticDepth(dropout_ratio, 'row') if self.has_skip else nn.Identity
        self.drop = nn.Dropout(dropout_ratio) if self.has_skip else nn.Identity
        # 不太懂是不是有stride和初始的没有连接在一起导致过拟合了

    def forward(self, input_tensor):
        x = input_tensor
        out = self.model(x)
        if self.has_skip:
            return input_tensor + self.stochastic_depth(out)
            # return self.drop(input_tensor) + out
        return out
