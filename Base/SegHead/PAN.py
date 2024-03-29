# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 14:10
# @Author  : ljq
# @desc    : 
# @File    : PAN.py
from torch import nn
import torch.nn.functional as F
from Base.MyModule.Seg import FPA, GAU, SPFPA, ReduceGAU
from Base.MyModule import SegHead, RegHead


class PANDecoder(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64], decoder_channel=128):
        super().__init__()
        if len(channels) == 4:
            channels.append(channels[-1])
        # 这边底层搞一个

        self.fpa = SPFPA(channels[0], decoder_channel)
        self.decoder0 = GAU(channels[1], decoder_channel)
        self.decoder1 = GAU(channels[2], decoder_channel)
        # 最后几层的channel减半
        # self.decoder2 = GAU(channels[3], decoder_channel)
        self.decoder2 = ReduceGAU(channels[3], decoder_channel // 2, decoder_channel)
        self.decoder3 = GAU(channels[4], decoder_channel // 2)
        self.last_channel = decoder_channel // 2
        self.channels = [decoder_channel, decoder_channel, decoder_channel, decoder_channel // 2, decoder_channel // 2]
        # self.init_weights()

    def get_stage(self):
        return [self.decoder0, self.decoder1, self.decoder2, self.decoder3]

    def forward(self, features):
        x = features[0]
        intern_features = features[1:]
        stages = self.get_stage()
        x = self.fpa(x)
        for idx, f, stage in zip(range(len(stages)), intern_features, stages):
            x = stage(x, f)
        # TODO： 因为是小目标，所以这边暂时使用两层的upsample
        # 最后对结果添加一个分类头
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def bottom_feature(self, bottom_feature):
        # 用来做底层特征加工
        return self.fpa(bottom_feature)

    def init_weights(self):
        # 初始化卷积和BatchNorm
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # layer.bias.data.fill_(0.001)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


class PAN(nn.Module):
    # 金子塔解码
    def __init__(self, encoder, decoder, seg_num, using_cls=False, reg_num=None, activation=''):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 这边加各种分类头
        self.seg = SegHead(decoder.last_channel, seg_num, activation)
        self.reg_num = reg_num
        if not reg_num is None:
            self.reg = RegHead(decoder.last_channel, reg_num)
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
