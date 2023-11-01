# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 11:12
# @Author  : ljq
# @desc    : 1. SMP中的Unet torch.cat 是不对等的。 2. SMP中的Head比较长 3. Unet论文中upconv后接两个卷积的结果会太深了 4.Unet底下的1024再upconv上来也是不必要的
# @File    : UnetHead.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class UPConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.UpConv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(output_channel * 2, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.UpConv(x)
        x = torch.cat([skip, x], dim=1)
        x = self.ConvBlock(x)
        return x


class BottomConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.ConvBlock(x)

        return x


class ClsBlock(nn.Module):
    def __init__(self, input_channel, cls, activation_type='sigmoid'):
        super().__init__()
        if activation_type == 'sigmoid':
            activation = nn.Sigmoid()
        elif cls == 'softmax':
            activation = nn.Softmax(dim=1)
        else:
            activation = nn.Sequential()

        # 加上一个以为卷积
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, cls, kernel_size=1, stride=1),
            activation,
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


class RegBlock(nn.Module):
    def __init__(self, input_channel):
        super(RegBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(input_channel, 1, kernel_size=1, stride=1))

    def forward(self, input_tensor):
        return self.block(input_tensor)


class UnetHead(nn.Module):
    def __init__(self, cls_num, channels=[512, 256, 128, 64], activation='sigmoid'):
        """
        :param channels: [512,256,128,64]
        :param cls_num: 分类的类别+1似乎
        """
        super().__init__()
        # 从Channels来选择
        self.decoder_layer1 = UPConv(channels[0], channels[1])
        self.decoder_layer2 = UPConv(channels[1], channels[2])
        self.decoder_layer3 = UPConv(channels[2], channels[3])
        # resnet的头和尾部是一样的
        self.decoder_layer4 = UPConv(channels[3], channels[3])

        # cls head
        self.cls = ClsBlock(channels[3], cls_num, activation)

    def get_stage(self):
        return [self.decoder_layer1, self.decoder_layer2, self.decoder_layer3, self.decoder_layer4]

    def forward(self, features):
        x = features[0]
        intern_features = features[1:]
        stages = self.get_stage()
        # 开始进行upsample
        for f, stage in zip(intern_features, stages):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = stage(x, f)
        # 最后对结果添加一个分类头
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.cls(x)


class Unet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor):
        features = self.encoder.feature_extract(input_tensor)
        x = self.decoder(features[::-1])
        return x
