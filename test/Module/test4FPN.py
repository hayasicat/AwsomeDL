# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 15:50
# @Author  : ljq
# @desc    : 
# @File    : test4FPN.py
import torch
from Base.SegHead.FPN import FPN, FPNDecoder
from Base.BackBone import ResNet34, ResNet18
from Base.SegHead.FPN import FPNAggregate

device = torch.device('cuda:0')
FPNAggregate()
encoder = ResNet34(20, small_scale=False)
decoder = FPNDecoder(encoder.channels)
model = FPN(encoder, decoder, 21)
model = model.to(device)
input_tensor = torch.ones(1, 3, 512, 512).to(device)
result = model(input_tensor)
print(result.size())
