# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 11:12
# @Author  : ljq
# @desc    : 
# @File    : FCNHead.py
import torch.nn as nn
import torch.nn.functional as F


class FCNDecoder(nn.Module):
    def __init__(self, input_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(input_channel, input_channel // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(input_channel // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(input_channel // 4, out_channel, 1)
        ])

    def forward(self, feature):
        return self.model(feature)


class FCN(nn.Module):
    def __init__(self, encoder, decoder, input_size=224):
        super(FCN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_size = input_size

    def forward(self, input_tensor, size=(224, 224)):
        features = self.encoder.feature_extract(input_tensor)
        feature = features[-1]
        feature = self.decoder(feature)
        # 上采样
        feature = F.interpolate(feature, size=size, mode='bilinear', align_corners=False)
        return feature
