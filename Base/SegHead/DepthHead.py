# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 10:08
# @Author  : ljq
# @desc    :  本仓库参考一下
# @File    : DepthHead.py
import torch
from torch import nn
import torch.nn.functional as F
from Base.SegHead.Unet import UPConv


def upsample(x):
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    return x


class Conv3x3(nn.Module):
    kernel_size = (3, 3)
    group = 1

    def __init__(self, input_channel, output_channel, stride=1, padding=1, act=nn.ReLU):
        super().__init__()
        # 修复Group没用的Bug
        self.conv = nn.Conv2d(input_channel, output_channel, self.kernel_size, stride=stride, padding=padding,
                              groups=self.group)
        self.bn = nn.BatchNorm2d(output_channel)
        if act is nn.ReLU:
            self.act = act(inplace=True)
        else:
            self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthUPConv(UPConv):
    def __init__(self, input_channel, output_channel, sec_input_channel=None):
        super(DepthUPConv, self).__init__(input_channel, output_channel, sec_input_channel=sec_input_channel)

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = upsample(x)
        # 在连在一块
        x = torch.cat([skip, x], dim=1)
        x = self.ConvBlock(x)
        return x


class LastDepthHead(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(LastDepthHead, self).__init__()
        self.conv1 = Conv3x3(input_channel, input_channel // 2, act=nn.ELU)
        self.conv2 = Conv3x3(input_channel // 2, output_channel, act=nn.Sigmoid)

    def forward(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = upsample(x)
        # 在连在一块
        x = self.conv2(x)
        return x


class Relu1(nn.Module):
    def __init__(self, inplace=True):
        super(Relu1, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input) / 6


class DepthDecoder(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], use_skips=True):
        super().__init__()
        # 这里不是一个标准的Unet结构，Unet是先upsample在Conv,这个是先Conv再UPSAMPLE
        self.encoder_channel = channels[::-1]
        self.decoder_channel = channels[::-1]
        # encoder 如果是大小如果是4的话，那么就添加一个相同的
        if len(self.encoder_channel) == 4:
            self.encoder_channel.append(self.encoder_channel[-1])
        # 因为还要上采样一次
        if len(self.decoder_channel) == 4:
            self.decoder_channel.append(self.decoder_channel[-1] // 2)
        self.decoder_channel.pop(0)

        # self.decoder_channel.append(self.decoder_channel[-1] // 2)

        self.decoder_layer1 = DepthUPConv(self.encoder_channel[0], self.decoder_channel[0])
        self.decoder_layer2 = DepthUPConv(self.encoder_channel[1], self.decoder_channel[1])
        self.decoder_layer3 = DepthUPConv(self.encoder_channel[2], self.decoder_channel[2])
        # resnet的头和尾部是一样的,就append一个新的进来
        self.decoder_layer4 = DepthUPConv(self.encoder_channel[3], self.decoder_channel[3],
                                          sec_input_channel=self.encoder_channel[4] + self.decoder_channel[3])
        # 输出多个深度的head
        self.depth_head1 = Conv3x3(self.decoder_channel[1], 1, act=nn.Sigmoid)
        self.depth_head2 = Conv3x3(self.decoder_channel[2], 1, act=nn.Sigmoid)
        self.depth_head3 = Conv3x3(self.decoder_channel[3], 1, act=nn.Sigmoid)
        self.depth_head4 = LastDepthHead(self.decoder_channel[3], 1)

    def get_decoder_stage(self):
        return [self.decoder_layer1, self.decoder_layer2, self.decoder_layer3, self.decoder_layer4]

    def get_depth_head(self):
        return [self.depth_head1, self.depth_head2, self.depth_head3, self.depth_head4]

    def forward(self, features):
        x = features[0]
        intern_features = features[1:]
        dec_stages = self.get_decoder_stage()
        depth_heads = self.get_depth_head()
        # 开始进行upsample
        disp_map = []

        for i, f, dec in zip(range(4), intern_features, dec_stages):
            x = dec(x, f)
            if i > 0:
                disp_map.append(depth_heads[i - 1](x))
        disp_map.append(depth_heads[-1](x))
        return disp_map


class DepthNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor):
        features = self.encoder.feature_extract(input_tensor)
        x = self.decoder(features[::-1])
        return x


if __name__ == "__main__":
    from Base.BackBone.ResNet import ResNet34
    from Base.BackBone.EfficientNetV2 import EfficientNetV2S

    encoder = EfficientNetV2S(10, input_chans=3)
    # encoder = ResNet34(10, input_chans=3)
    de = DepthDecoder(encoder.channels)
    d = DepthNet(encoder, de)
    input_tensor = torch.ones((1, 3, 640, 640))
    outs = d(input_tensor)

    for o in outs:
        print(o.size())
