# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 14:16
# @Author  : ljq
# @desc    :  只是验证一下流程
# @File    : SimpleCnn.py
import torch
import torch.nn as nn


class SimpleCnn(nn.Module):
    def __init__(self, num_cls, input_chans=3):
        super().__init__()
        # 简单两层的cnn
        model = [
            nn.Conv2d(input_chans, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
        self.model = nn.Sequential(*model)
        # 简单一层dnn
        self.cls = nn.Sequential(*[
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])
        self.fc = nn.Linear(512, num_cls)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.cls(x)
        return self.fc(x)

    def init_weights(self):
        # 初始化卷积和BatchNorm
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # layer.bias.data.fill_(0.001)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
