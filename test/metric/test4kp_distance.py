# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 9:56
# @Author  : ljq
# @desc    : 
# @File    : test4kp_distance.py
import torch
import numpy as np
from Base.Metrics.KP import KPDis

matrix = torch.randn((7, 7))
gts = np.array([4, 4])
print(matrix, gts)
