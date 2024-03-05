# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 14:10
# @Author  : ljq
# @desc    : 
# @File    : PAN.py
from torch import nn
import torch.nn.functional as F
from Base.MyModule.Seg import FPA, GAU
from Base.MyModule import SegHead


class PANDecoder(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64], decoder_channel=128):
        super().__init__()
        if len(channels) == 4:
            channels.append(channels[-1])
        # 这边底层搞一个

        self.fpa = FPA(channels[0], decoder_channel)
        self.decoder0 = GAU(channels[1], decoder_channel)
        self.decoder1 = GAU(channels[2], decoder_channel)
        self.decoder2 = GAU(channels[3], decoder_channel)
        self.decoder3 = GAU(channels[4], decoder_channel)
        self.last_channel = channels[-1]

    def get_stage(self):
        return [self.decoder_layer1, self.decoder_layer2, self.decoder_layer3, self.decoder_layer4]

    def forward(self, features):
        x = features[0]
        intern_features = features[1:]
        stages = self.get_stage()
        x = self.fpa(x)
        for f, stage in zip(intern_features, stages):
            x = stage(x, f)
        # TODO： 因为是小目标，所以这边暂时使用两层的upsample
        # 最后对结果添加一个分类头
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class PAN(nn.Module):
    # 金子塔解码
    def __init__(self, encoder, decoder, seg_num, using_cls=False, activation=''):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 这边加各种分类头
        self.seg = SegHead(decoder.last_channel, seg_num, activation)
        self.using_cls = using_cls

    def forward(self, input_tensor):
        features = self.encoder.feature_extract(input_tensor)
        x = self.decoder(features[::-1])
        # 修改了一下
        return_info = []
        return_info.append(self.seg(x))
        if not self.reg_num is None:
            return_info.append(self.reg(x))
        # 返回结果
        if self.using_cls:
            # 增加抓想和叠箱两种任务的区分
            cls = self.encoder.get_cls(features[-1])
            return_info.append(cls)

        return return_info
