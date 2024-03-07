# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 9:24
# @Author  : ljq
# @desc    : 
# @File    : pose_trian.py

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from thop import profile

from VisualTask.Seg.trainner import SegTrainner
from VisualTask.Seg.multi_head import SegMultiHead
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.SegHead.FPN import FPN, FPNDecoder
from Base.SegHead.PAN import PANDecoder, PAN

from Base.BackBone import ResNet34, ResNet18, EfficientNetV2S, TorchvisionResnet18
from Transfer.VisualFLS.dataset import FLSHybridDataset, FLS_norm_transform, FLS_test_transforms, \
    FLS_train_transforms_kp

device = torch.device('cuda')

train_dataset = FLSHybridDataset('/root/data/VisualFLS', is_crop=False, transforms=FLS_train_transforms_kp,
                                 norm_transforms=FLS_norm_transform)
val_dataset = FLSHybridDataset('/root/data/VisualFLS', is_crop=False, is_train=False, transforms=FLS_test_transforms,
                               norm_transforms=FLS_norm_transform)

# root_path = '../../data/lockhole/multi_head/UnetTotal'
root_path = '../../data/lockhole/multi_head/torchPan_TC4'

if not os.path.exists(root_path):
    os.makedirs(root_path)
# encoder = EfficientNetV2S(20)
encoder = TorchvisionResnet18(2)
# encoder = ResNet34(20, small_scale=False)
# decoder = UnetHead(encoder.channels[::-1])
# 如果使用的是focal loss这边应该是sigmoid
# model = Unet(encoder, decoder, 3, reg_num=2, using_cls=True, activation='')

# decoder = FPNDecoder(encoder.channels)
# model = FPN(encoder, decoder, 3, reg_num=2, activation='')

decoder = PANDecoder(encoder.channels[::-1])
model = PAN(encoder, decoder, 3, reg_num=2, using_cls=True, activation='')

# flops, params = profile(model, (torch.randn(2, 3, 512, 512),))
# print("flops is {},params is {}".format(flops, params))

trainner = SegMultiHead(train_dataset, val_dataset, model, 3, root_path, lr=0.0001, batch_size=12)
trainner.train()
