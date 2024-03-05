# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 10:01
# @Author  : ljq
# @desc    : 
# @File    : test4my_module.py
import torch

from Base.MyModule import FPA, GAU

input_tensor = torch.ones((10, 256, 32, 32))
encoder_tensor = torch.ones(10, 512, 64, 64)
fpa = FPA(256, 64)
print(fpa(input_tensor).size())
gau = GAU(512, 256)
up_tensor = gau(input_tensor, encoder_tensor)
print(up_tensor.size())
