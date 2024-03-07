# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 16:47
# @Author  : ljq
# @desc    : 
# @File    : EDShortCut.py
import torch

from torch import nn
import torch.nn.functional as F
from ..BasicModule import ConvBnAct, ChannelPool
from ..BasicModule.ConvLayer import ConvBn


class GAU(nn.Module):
    # Pyramid Attention Network
    def __init__(self, input_channel, output_channel, **kwargs):
        super(GAU, self).__init__()
        self.decoder_branch = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnAct(output_channel, output_channel, kernel_size=1, padding=0, act=nn.Sigmoid)
        ])
        # 更改一下1x1的卷积吧
        self.encoder_branch = ConvBnAct(input_channel, output_channel, kernel_size=3)

    def forward(self, x, skip):
        # 这里用
        intern_feature = torch.mul(self.decoder_branch(x), self.encoder_branch(skip))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return intern_feature + x


class ReduceGAU(GAU):
    def __init__(self, input_channel, output_channel, dec_channel):
        super().__init__(input_channel, output_channel)
        self.reduce_conv = ConvBnAct(dec_channel, output_channel, kernel_size=3)

    def forward(self, x, skip):
        x = self.reduce_conv(x)
        intern_feature = torch.mul(self.decoder_branch(x), self.encoder_branch(skip))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return intern_feature + x


class DownGAU(nn.Module):
    def __init__(self, enc_channel, dec_channel, output_channel):
        super(DownGAU, self).__init__()
        self.point_conv = ConvBnAct(dec_channel, output_channel, kernel_size=1, padding=0)
        self.decoder_branch = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnAct(output_channel, output_channel, kernel_size=1, padding=0, act=nn.Sigmoid)
        ])
        self.encoder_branch = ConvBnAct(enc_channel, output_channel, kernel_size=3)

    def forward(self, x, skip):
        # 这里用
        x = self.point_conv(x)
        intern_feature = torch.mul(self.decoder_branch(x), self.encoder_branch(skip))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return intern_feature + x


class FPA(nn.Module):
    # Feature Pyramid Attention
    # 作者大概是下采样了，然后中间的卷积当作是CBAM中的通道注意力吧
    def __init__(self, input_channel, output_channel):
        super(FPA, self).__init__()
        ratio = 0.25
        self.GP = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnAct(input_channel, output_channel, act=nn.ReLU)
        ])
        self.mid_branch = ConvBnAct(input_channel, output_channel, kernel_size=1, padding=0)
        # 降采样
        self.down1 = ConvBnAct(input_channel, int(ratio * input_channel), kernel_size=3, stride=2)
        self.down2 = ConvBnAct(int(ratio * input_channel), int(ratio * input_channel), kernel_size=3, stride=2)
        self.down3 = ConvBnAct(int(ratio * input_channel), int(ratio * input_channel), kernel_size=3, stride=2)

        # 这边使用shortcut+reduce/或者这边可以使用共享的卷积
        self.reduce_conv1 = ConvBn(2, 1, kernel_size=3, stride=1)
        self.reduce_conv2 = ConvBn(2, 1, kernel_size=3, stride=1)
        self.reduce_conv3 = ConvBn(2, 1, kernel_size=3, stride=1)

        self.spatial_pool = ChannelPool()

    def forward(self, x):
        b, c, h, w = x.size()
        gp = self.GP(x)
        gp = F.interpolate(gp, (h, w), mode='bilinear', align_corners=True)
        # 中间的一层
        mid = self.mid_branch(x)
        # 底下空间注意力那一层
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        # 降低通道
        u3 = self.reduce_conv3(self.spatial_pool(d3))
        u2 = self.reduce_conv2(self.spatial_pool(d2))
        u1 = self.reduce_conv1(self.spatial_pool(d1))
        # 相加以后来一个sigmoid
        sam = F.interpolate(u3, scale_factor=8, mode='bilinear', align_corners=True) + \
              F.interpolate(u2, scale_factor=4, mode='bilinear', align_corners=True) + \
              F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=True)
        # 注意力机制？
        x = torch.mul(torch.sigmoid(sam), mid)
        return x + gp


class SPFPA(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SPFPA, self).__init__()
        ratio = 0.25
        self.GP = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBnAct(input_channel, output_channel, act=nn.ReLU)
        ])
        self.mid_branch = ConvBnAct(input_channel, output_channel, kernel_size=1, padding=0)

        self.down1 = ConvBnAct(input_channel, int(ratio * input_channel), kernel_size=3, stride=1, act=nn.ELU)

        self.reduce_conv = ConvBn(2, 1, kernel_size=3, stride=1, act=nn.ELU)
        self.spatial_pool = ChannelPool()

    def forward(self, x):
        gp = self.GP(x)
        # 中间的一层
        mid = self.mid_branch(x)
        # 底下空间注意力那一层
        d1 = self.down1(x)
        # 不降低通道直接来一个空间注意力
        u1 = self.reduce_conv(self.spatial_pool(d1))
        sam = u1
        # sam = F.interpolate(sam, scale_factor=2, mode='bilinear', align_corners=True)
        mid = torch.mul(torch.sigmoid(sam), mid)
        return mid + gp
