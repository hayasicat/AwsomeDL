# -*- coding: utf-8 -*-
# @Time    : 2024/2/6 8:56
# @Author  : ljq
# @desc    : 
# @File    : TochvisionBackbone.py
import torch
from torch import nn
from torchvision.models.resnet import ResNet, model_urls, resnet18, resnet34
from torchvision.models.efficientnet import efficientnet_b0


# 继承自torch Vision的resnet模型


class TorchvisionResnet18(nn.Module):
    # 用来做预训练的backbone来使用

    def __init__(self, num_cls, input_chans=3, pretrain=True):
        super(TorchvisionResnet18, self).__init__()
        self.model = self.init_model(pretrained=pretrain)
        # 如果输入的通道不是3的话就替换一下
        if input_chans != 3:
            self.model.conv1 = nn.Conv2d(input_chans, self.model.inplanes, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        # 如果输出的类别不是１０００的话
        if num_cls != 1000:
            self.model.fc = nn.Linear(512, num_cls)
        # 通用的配置信息
        self.channels = [64, 128, 256, 512]

    def get_stages(self):
        return [nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu),
                nn.Sequential(self.model.maxpool, self.model.layer1), self.model.layer2, self.model.layer3,
                self.model.layer4]

    def forward(self, input_tensor):
        return self.model.forward(input_tensor)

    def feature_extract(self, x):
        layers = self.get_stages()
        intern_features = []
        for layer in layers:
            x = layer(x)
            intern_features.append(x)
        return intern_features

    def get_cls(self, x):
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # 解决这个问题
        return x

    def init_model(self, **kwargs):
        return resnet18(**kwargs)


class TorchvisionResnet34(TorchvisionResnet18):
    def __init__(self, num_cls, input_chans=3, pretrain=True):
        # 初始化的函数给包一个
        super().__init__(num_cls, input_chans, pretrain)

    def init_model(self, **kwargs):
        return resnet34(**kwargs)


if __name__ == "__main__":
    model = TorchvisionResnet18(100)
    input_tensor = torch.ones((1, 3, 512, 512))
    print(model(input_tensor))
