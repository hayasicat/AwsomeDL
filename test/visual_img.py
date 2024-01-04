# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 11:30
# @Author  : ljq
# @desc    : 
# @File    : visual_img.py
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

import torch

from VisualTask.Seg.trainner import SegTrainner
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18
from Transfer.VisualFLS.dataset import FLSDataset, FLS_norm_transform, FLS_test_transforms, FLS_train_transforms
import matplotlib.pyplot as plt

device = torch.device('cuda')

tansformer_dataset = FLSDataset('/backup/VisualFLS', is_crop=False, transforms=FLS_train_transforms)
or_dataset = FLSDataset('/backup/VisualFLS', is_crop=False)

# fig, axes = plt.subplots(4, 3)
# print(axes)
for i in range(15):
    t_img, t_seg = tansformer_dataset[i]
    o_img, o_seg = or_dataset[i]
    # 切割成两个图
    rank = math.floor(i / 3.0)
    col = i % 3
    plt.subplot(121)
    # axes[rank, col].imshow(t_img.transpose(1, 2, 0))
    plt.imshow(t_img.transpose(1, 2, 0))
    # axes[rank + 2, col].imshow(o_img.transpose(1, 2, 0))
    plt.subplot(122)
    plt.imshow(o_img.transpose(1, 2, 0))
    plt.show()
