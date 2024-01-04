# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 17:08
# @Author  : ljq
# @desc    : 训练检测一个锁孔
# @File    : test4lockedhole.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

import torch

from VisualTask.Seg.trainner import SegTrainner
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18
from Transfer.VisualFLS.dataset import FLSDataset, FLS_norm_transform, FLS_test_transforms, FLS_train_transforms

device = torch.device('cuda')

train_dataset = FLSDataset('/backup/VisualFLS', is_crop=False, transforms=FLS_train_transforms,
                           norm_transforms=FLS_norm_transform)
val_dataset = FLSDataset('/backup/VisualFLS', is_crop=False, is_train=False, transforms=FLS_test_transforms,
                         norm_transforms=FLS_norm_transform)

root_path = '../data/lockhole/small_picture'
if not os.path.exists(root_path):
    os.makedirs(root_path)

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

encoder = ResNet34(20, small_scale=False)
decoder = UnetHead()
# head应该在这边
model = Unet(encoder, decoder, 3, activation='')

trainner = SegTrainner(train_dataset, val_dataset, model, 3, save_path=root_path, lr=0.0001)
trainner.batch_size = 12
trainner.train()
