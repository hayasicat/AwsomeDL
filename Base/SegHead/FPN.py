# -*- coding: utf-8 -*-
# @Time    : 2023/11/14 14:22
# @Author  : ljq
# @desc    : 手搓一个FPN，
# @File    : FPN.py
import torch
from torch import nn
import torch.nn.functional as  F

from .Unet import ClsBlock, RegBlock


class ConvBnRelu(nn.Module):
    def __init__(self, kernel_size, input_channel, output_channel):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=int((kernel_size - 1) // 2)),
            # 为什么这里不要BatchNormal
            #nn.BatchNorm2d(output_channel),
            #nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        return self.model(x)


class ConvGnRelu(nn.Module):
    def __init__(self, kernel_size, input_channel, output_channel, isUpsample=True, group=32):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=int(kernel_size - 1) // 2),
            nn.GroupNorm(group, output_channel),
            nn.ReLU(inplace=True)
        ])
        self.isUpsample = isUpsample

    def forward(self, x):
        x = self.model(x)
        if self.isUpsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class TopDownBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(TopDownBlock, self).__init__()
        self.skip_block = ConvBnRelu(1, input_channel, output_channel)

    def forward(self, x, skip):
        # skip经过1x1卷积以后在计算
        skip = self.skip_block(skip)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return skip + x


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channel, feature_channel=256):
        # 因为用了BN以后就不想使用Dropout
        super().__init__()
        encoder_channel = encoder_channel[::-1]
        self.P5 = ConvBnRelu(1, encoder_channel[0], feature_channel)
        self.P4 = TopDownBlock(encoder_channel[1], feature_channel)
        self.P3 = TopDownBlock(encoder_channel[2], feature_channel)
        self.P2 = TopDownBlock(encoder_channel[3], feature_channel)

    def get_stage(self):
        return [self.P5, self.P4, self.P3, self.P2]

    def feature_extract(self, encoder_features):
        stages = self.get_stage()
        x = stages[0](encoder_features[0])
        decoder_features = []
        decoder_features.append(x)
        for skip, stage in zip(encoder_features[1:], stages[1:]):
            x = stage(x, skip)
            decoder_features.append(x)
        return decoder_features


class FPNAggregate(nn.Module):
    def __init__(self, input_channel=256, output_channel=128,dropout=0.2):
        super().__init__()
        upsample_times = [3, 2, 1, 0]
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.P5S = self.make_sematic_branches(upsample_times[0])
        self.P4S = self.make_sematic_branches(upsample_times[1])
        self.P3S = self.make_sematic_branches(upsample_times[2])
        self.P2S = self.make_sematic_branches(upsample_times[3])
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def make_sematic_branches(self, upsample_time):
        c_branch = [ConvGnRelu(3, self.input_channel, self.output_channel, bool(upsample_time))]
        for n in range(0, upsample_time - 1, 1):
            c_branch.append(ConvGnRelu(3, self.output_channel, self.output_channel))
        return nn.Sequential(*c_branch)

    def get_stage(self):
        return [self.P5S, self.P4S, self.P3S, self.P2S]

    def forward(self, decoder_features):
        semantic_features = []
        stages = self.get_stage()
        for f, stage in zip(decoder_features, stages):
            semantic_features.append(stage(f))
        x = sum(semantic_features)
        return self.dropout(x)


class FPN(nn.Module):
    def __init__(self, encoder, decoder, cls_num, reg_num=None, activation=''):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.aggregate = FPNAggregate()
        # cls head
        self.cls = ClsBlock(128, cls_num, activation)
        self.reg_num = reg_num
        if not reg_num is None:
            self.reg = RegBlock(128, reg_num)

    def forward(self, input_tensor):
        features = self.encoder.feature_extract(input_tensor)
        decoder_features = self.decoder.feature_extract(features[::-1])
        semantic_feature = self.aggregate(decoder_features)
        # 加HEAD，加结果
        cls_pred = self.cls(semantic_feature)
        cls_pred = F.interpolate(cls_pred, scale_factor=4, mode='bilinear', align_corners=False)
        if self.reg_num is None:
            return cls_pred
        reg_pred = self.reg(semantic_feature)
        reg_pred = F.interpolate(reg_pred, scale_factor=4, mode='bilinear', align_corners=False)
        return cls_pred, reg_pred
