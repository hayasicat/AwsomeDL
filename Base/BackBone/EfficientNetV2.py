# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 17:44
# @Author  : ljq
# @desc    : 参考
# 1. https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py#L270
# 2. https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
# @File    : EfficientNet.py
# TODO： 为什么Mobilenet需要有一下Dropout这种东西，一般来说有BN不就可以抛弃DropOut了吗
# TODO： 小数据集上大的batchsize 容易过拟合，还是小batchsize慢慢训比较稳

import torch
from torch import nn
from torchvision.ops.stochastic_depth import StochasticDepth

from ._utils import Conv3x3, DepthWiseConv, DPWConv, PointWiseConv
from ._utils import SELayer


class MBConv(nn.Module):
    def __init__(self, input_channel, output_channel, expand_ratio=4, stride=1, has_skip=True, dropout_ratio=0.0):
        super().__init__()
        inter_media_channel = input_channel * expand_ratio
        self.has_skip = has_skip
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


class ClassHead(nn.Module):
    def __init__(self, in_ch, out_ch, num_cls, dropout_ratio=0.1):
        super(ClassHead, self).__init__()
        # TODO: 使用常规的head
        self.num_cls = num_cls
        self.head = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            PointWiseConv(in_ch, out_ch, act=nn.ReLU)
        ])
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(in_ch, out_ch)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(out_ch, num_cls)

    def forward(self, x):

        x = self.head(x)
        x = self.dropout(x)
        return self.fc2(x)


class EfficientNetV2S(nn.Module):
    def __init__(self, num_class, input_chans=3, max_drop_ratio=0.1):
        super(EfficientNetV2S, self).__init__()
        # 加了很多MBConv还不如直接去掉只有FuseMBConv,超参的dropout率设置在0.5附近。因此如何使得MBCONV发挥作用
        # 实验数据 FuseMBConv 256最终输出结果为，可能为最后一层梯度太深没办法传递回去80->0.54
        # stage3的时候模型也是很难收敛，精度也是只有0.47，要减少一下Overfitting
        # 如果数据是小数据的话，要更大的dropout以及说更大的
        # 如果太深很容易影响到最终的网络精度。dropout这种在太深的网络的影响性较少
        # self.channels = [24, 24, 48, 64, 128, 160, 256]
        self.block_channels = [24, 24, 48, 64, 128, 160, 256]
        # 一共有五层，五层的channel就不一样
        self.channels = [24, 48, 64, 128, 256]
        # 根据数据集的性质来选定block_num 上面这个是cifar100的较优参数
        # self.block_num = [2, 4, 2, 3, 4, 5]
        self.block_num = [2, 4, 4, 6, 9, 15]
        # 拟合能力比较差的话给后面的最后一层来点惊喜
        self.total_block = sum(self.block_num[:4])
        self.current_block = 0
        self.max_drop_ratio = max_drop_ratio
        # self.block_num = [1, 2, 2, 3, 3, 1]

        self.expand_ratios = [1, 4, 4, 4, 6, 6]
        self.strides = [1, 2, 2, 2, 2, 1]
        # 小图片去掉stride试试
        self.ConvStem = Conv3x3(input_chans, self.block_channels[0], stride=2)
        self.stage1 = self.make_block(self.block_channels[:2], self.block_channels[1:3], self.block_num[:2],
                                      self.expand_ratios[:2],
                                      self.strides[:2], FuseMBConv)
        self.stage2 = self.make_block([self.block_channels[2]], [self.block_channels[3]], [self.block_num[2]],
                                      [self.expand_ratios[2]],
                                      [self.strides[2]], FuseMBConv)
        self.stage3 = self.make_block([self.block_channels[3]], [self.block_channels[4]], [self.block_num[3]],
                                      [self.expand_ratios[3]],
                                      [self.strides[3]], MBConv)
        self.stage4 = self.make_block(self.block_channels[4:6], self.block_channels[5:], self.block_num[4:6],
                                      self.expand_ratios[4:6],
                                      self.strides[4:6], MBConv)
        self.cls_head = ClassHead(self.block_channels[-1], 1280, num_class)

    def make_block(self, input_channels, output_channels, repeat_times, expand_ratios, strides, BlockFunc: FuseMBConv):
        layers = []
        for in_ch, out_ch, repeat, ex_ra, s in zip(input_channels, output_channels, repeat_times, expand_ratios,
                                                   strides):
            for i in range(repeat):
                has_skip = True
                if s > 1:
                    has_skip = False
                if in_ch != out_ch:
                    has_skip = False
                dropout_ratio = self.get_drop_ratio()

                layers.append(BlockFunc(in_ch, out_ch, ex_ra, s, has_skip, dropout_ratio=dropout_ratio))
                in_ch = out_ch
                s = 1
        return nn.Sequential(*layers)

    def get_stages(self):
        return [self.ConvStem, self.stage1, self.stage2, self.stage3, self.stage4]

    def feature_extract(self, x):
        """
        作为encoder的时候输入
        :param x:
        :return:
        """
        layers = self.get_stages()
        intern_features = []
        for layer in layers:
            x = layer(x)
            intern_features.append(x)
        return intern_features

    def get_drop_ratio(self):
        drop_ratio = (self.current_block / float(self.total_block)) * self.max_drop_ratio
        self.current_block += 1
        return drop_ratio

    def forward(self, input_tensor, **kwargs):
        x = self.ConvStem(input_tensor)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.cls_head(x)

    def init_weights(self):
        # 初始化卷积和BatchNorm
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # layer.bias.data.fill_(0.001)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


if __name__ == "__main__":
    net = EfficientNetV2S(10)
    test_tensor = torch.ones((1, 3, 640, 640))
    result = net(test_tensor)
    print(result.size())
