# _*_ coding:utf-8 _*_
# @author:ljq
# @date: 2023/8/2 23:32
# @TODO： 1. 创建模型的代码有问题，不收敛，配置为resnet18,过拟合了，相同的代码放到torchvison.models反倒是不会的  (更改了repeats以及基础的模块以后反而好了)
# @差异： 1. BasicBlock中第二个卷积没有nn.relu  -> 原因不在于此，加上以后精度也没有发生变化
#         2. 去掉MaxPool2d让第一个block来训练   -> 没有用      -> 对于小图片来说MaxPool2d还是掉点蛮严重的
#          3. Conv去掉bias                     -> 似乎能panelty了
#          4. 首层的strid变成1                  -> 没有用的    -> 对于小图片来说strid为2也是掉点蛮严重的
#          5. 首层的kernel size变成3            -> 也是没有用
#          6. 用.vew代替flantten                -> 没什么效果
#          7. 首层的padding 改为1               -> 似乎效果是不错的
# 原因可能是出现在首层，小图片分类会有问题。

import torch
from torch import nn


class BasicBlock(nn.Module):
    # 基于34层以下的模块
    def __init__(self, input_channel, output_channel, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 1
        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, (1, 1), stride=2),
                nn.BatchNorm2d(output_channel)
            )
            stride = 2
        model = []
        strides = [stride, 1]
        for i, s in enumerate(strides):
            model.append(nn.Conv2d(input_channel, output_channel, (3, 3), stride=s, padding=1, bias=False))
            model.append(nn.BatchNorm2d(output_channel))
            if i == 0:
                model.append(nn.ReLU(inplace=True))
            input_channel = output_channel
        self.model = nn.Sequential(*model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.model(input)
        x = self.shortcut(input)
        return self.relu(out + x)


class BottleneckBlock(nn.Module):
    def __init__(self, input_channel, output_channel, downsample=False):
        super().__init__()
        stride = 1
        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, (1, 1), stride=2),
                nn.BatchNorm2d(output_channel),
                # 这边添加一下Maxpool2D
            )
            stride = 2

        model = []
        kernel_sizes = [1, 3, 1]
        feature_maps = [int(output_channel / 4), int(output_channel / 4), output_channel]
        for i, k, f in zip(range(3), kernel_sizes, feature_maps):
            padding = 0
            if k == 3:
                padding = 1
            model.append(nn.Conv2d(input_channel, f, (k, k), stride=stride, padding=padding))
            model.append(nn.BatchNorm2d(f))
            model.append(nn.ReLU(inplace=True))
            input_channel = f
            stride = 1

        self.model = nn.Sequential(*model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        out = self.model(input)
        x = self.shortcut(x)

        return self.relu(out + x)


def make_block(baseblock, input_chan, output_chan, repeat, need_downsample=False):
    layer = []
    downsamples = [need_downsample] + (repeat - 1) * [False]
    for i, d in zip(range(repeat), downsamples):
        layer.append(baseblock(input_chan, output_chan, d))
        input_chan = output_chan
    return nn.Sequential(*layer)


class ResNet(nn.Module):
    def __init__(self, BaseBlock, num_class, input_chans=3, repeats=[2, 2, 2, 2], small_scale=False):
        super().__init__()
        # 不同层的话就是配置的内容不一样
        channels = [64, 128, 256, 512]
        if small_scale:
            self.first_conv = nn.Conv2d(input_chans, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.first_conv = nn.Conv2d(input_chans, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.ConvHead = nn.Sequential(
            self.first_conv,
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        # 要兼容小图片的分类以及打图片的分类
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 如果是列表的就没办法直接转到gpu上去了
        self.layer1 = make_block(BaseBlock, channels[0], channels[0], repeats[0], False)
        self.layer2 = make_block(BaseBlock, channels[0], channels[1], repeats[1], True)
        self.layer3 = make_block(BaseBlock, channels[1], channels[2], repeats[2], True)
        self.layer4 = make_block(BaseBlock, channels[2], channels[3], repeats[3], True)
        # 如果在init加入列表的话，并行就有问题
        # 添加全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_class)

    def get_stages(self):
        return [self.ConvHead, nn.Sequential(self.maxpool2d, self.layer1), self.layer2, self.layer3, self.layer4]

    def forward(self, input_tensor):
        x = self.ConvHead(input_tensor)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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

    def init_weights(self):
        # 初始化卷积和BatchNorm
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # layer.bias.data.fill_(0.001)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


class ResNet18(ResNet):
    def __init__(self, num_cls):
        super().__init__(BasicBlock, num_cls)


class ResNet34(ResNet):
    def __init__(self, num_cls, small_scale=False):
        super().__init__(BasicBlock, num_cls, repeats=[3, 4, 6, 3], small_scale=small_scale)


class ResNet50(ResNet):
    def __init__(self, num_cls):
        #
        super().__init__(BottleneckBlock, num_cls)


if __name__ == '__main__':
    # a = BasicBlock(128, 64, is_downsample=True)
    # a.init_weights()
    # t = a(torch.zeros(1, 128, 64, 64))
    # print(t.size())
    m = ResNet(BottleneckBlock, 100)
    m.init_weights()
    t = m(torch.zeros(1, 3, 224, 224))
    print(t.size())
