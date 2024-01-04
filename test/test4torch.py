# -*- coding: utf-8 -*-
# @Time    : 2023/11/23 17:35
# @Author  : ljq
# @desc    : 
# @File    : test4torch.py

import torch

range_vec_q = torch.arange(5)
range_vec_k = torch.arange(5)
distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
print(distance_mat)