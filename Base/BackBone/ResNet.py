# _*_ coding:utf-8 _*_
# @author:ljq
# @date: 2023/8/2 23:32
import torch
from torch import nn


class BasicBlock(nn.Module):
    # 基于34层以下的模块
    def __init__(self, input_channel, output_channel, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 1
        if downsample:
            self.identity = nn.Conv2d(input_channel, output_channel, (1, 1), stride=2)
            stride = 2

        model = nn.Sequential()
        for i in range(2):
            stride = stride
            if i != 0:
                stride = 1
            model.append(nn.Conv2d(input_channel, output_channel, (3, 3), stride=stride, padding=1))
            model.append(nn.BatchNorm2d(output_channel))
            model.append(nn.ReLU(inplace=True))
            input_channel = output_channel

        self.model = model

    def forward(self, input):
        x = input
        if self.downsample:
            x = self.identity(x)
        input = self.model(input)

        return input + x


class BottleneckBlock(nn.Module):
    def __init__(self, input_channel, output_channel, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 1
        if downsample:
            self.identity = nn.Conv2d(input_channel, output_channel, (1, 1), stride=2)
            stride = 2

        model = nn.Sequential()
        kernal_sizes = [1, 3, 1]
        feature_maps = [int(output_channel / 4), int(output_channel / 4), output_channel]
        for i, k, f in zip(range(3), kernal_sizes, feature_maps):
            stride = stride
            if i != 0:
                stride = 1
            padding = 0
            if k == 3:
                padding = 1
            model.append(nn.Conv2d(input_channel, f, (k, k), stride=stride, padding=padding))
            model.append(nn.BatchNorm2d(f))
            model.append(nn.ReLU(inplace=True))
            input_channel = f

        self.model = model

    def forward(self, input):
        x = input
        if self.downsample:
            x = self.identity(x)
        input = self.model(input)

        return input + x


def make_block(baseblock, input_chans, repeats):
    base = nn.Sequential()
    input_chan = input_chans[0]
    for idx, chans, repeat in zip(range(len(input_chans)), input_chans, repeats):
        # 重复多次

        downsample = True
        if idx == 0:
            downsample = False
        for i in range(repeat):
            base.append(baseblock(input_chan, chans, downsample))
            downsample = False
            input_chan = chans
    return base


class ResNet(nn.Module):
    def __init__(self, BaseBlock, num_class, input_chans=3, input_size=256):
        super().__init__()
        # 不同层的话就是配置的内容不一样
        channels = [64, 128, 256, 512]
        repeats = [2, 2, 2, 2]
        self.conv1 = nn.Conv2d(input_chans, channels[0], kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.ReLU(inplace=True)
        # 中间层降采样
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 创建block不同size的模型重复的次数不变
        self.base = make_block(BaseBlock, channels, repeats)
        # 添加全连接层
        feature_size = int(input_size / 32)
        self.avgpool = nn.AvgPool2d(kernel_size=(feature_size, feature_size), stride=1)
        self.fc = nn.Linear(channels[-1], num_class)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.base(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weights(self):
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                layer.bias.data.fill_(0.001)


class ResNet34(ResNet):
    def __init__(self, num_cls, input_size):
        super().__init__(BottleneckBlock, num_cls, input_size=input_size)


if __name__ == '__main__':
    # a = BasicBlock(128, 64, is_downsample=True)
    # a.init_weights()
    # t = a(torch.zeros(1, 128, 64, 64))
    # print(t.size())
    m = ResNet(BottleneckBlock, 100, input_size=224)
    m.init_weights()
    t = m(torch.zeros(1, 3, 224, 224))
    print(t.size())
