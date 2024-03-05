# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 17:05
# @Author  : ljq
# @desc    : 
# @File    : MonoDepth.py
import torch
from torch import nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # 修正一下深度的head需要这样子
        self.conv = nn.Sequential(*[
            nn.Conv2d(input_channel, output_channel, (3, 3), padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        """
        就直接upsample以后来个conv
        :param x:
        :return:
        """
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)


class MonoDepth(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        # 这边添加不同的深度Head
        self.depth_head1 = DepthHead(self.decoder.channels[1], 1)
        self.depth_head2 = DepthHead(self.decoder.channels[2], 1)
        self.depth_head3 = DepthHead(self.decoder.channels[3], 1)
        self.depth_head4 = DepthHead(self.decoder.channels[4], 1)

    def get_depth_heads(self):
        return [self.depth_head1, self.depth_head2, self.depth_head3, self.depth_head3]

    def forward(self, input_tensor):
        # 得到encoder的特征图
        features = self.encoder.feature_extract(input_tensor)[::-1]
        # features
        bottom_feature = features[0]
        x = self.decoder.bottom_feature(bottom_feature)
        # 每一层直接添加
        stages = self.decoder.get_stage()
        intern_features = []
        for idx, f, stage in zip(range(len(stages)), features[1:], stages):
            x = stage(x, f)
            intern_features.append(x)
        # 每一层都进行上采样
        depth_maps = []
        depth_module = self.get_depth_heads()
        for f, head in zip(intern_features, depth_module):
            depth_maps.append(head(f))
        return depth_maps


if __name__ == "__main__":
    # 要用Pan模型来怼单目深度估计
    from Base.BackBone.ResNet import ResNet34
    from Base.BackBone.EfficientNetV2 import EfficientNetV2S
    from Base.SegHead.PAN import PANDecoder

    enc = EfficientNetV2S(10, input_chans=3)
    # encoder = ResNet34(10, input_chans=3)
    de = PANDecoder(enc.channels[::-1])
    d = MonoDepth(enc, de)
    input_tensor = torch.ones((3, 3, 640, 640))
    outs = d(input_tensor)

    for o in outs:
        print(o.size())
