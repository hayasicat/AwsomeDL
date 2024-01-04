# -*- coding: utf-8 -*-
# @Time    : 2023/12/23 23:21
# @Author  : ljq
# @desc    : 
# @File    : test4stoch.py
import torch

noise = torch.empty(100)
noise = noise.bernoulli_(0.1).div_(0.1)
print(noise)
