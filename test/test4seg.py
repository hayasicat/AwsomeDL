# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 9:13
# @Author  : ljq
# @desc    : 
# @File    : test4seg.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'

import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import albumentations as alb
from PIL import Image

from VisualTask.Seg.trainner import SegTrainner
from VisualTask.Seg.multi_head import SegMultiHead
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.SegHead import FCNDecoder, FCN
from Base.SegHead.FPN import FPN, FPNDecoder
from Base.BackBone import ResNet34, ResNet18
from Base.Dataset import MyVocDataset

device = torch.device('cuda')

# train_transformer = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(224),
#     torchvision.transforms.RandomCrop(224, padding=4),
#     torchvision.transforms.RandomHorizontalFlip(p=0.5),
#
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
#                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
#     # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# test_transformer = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
#                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
#
# ]
# )

# 这边有RandomResize
train_transformer = alb.Compose([
    alb.SmallestMaxSize(max_size=384),
    alb.RandomCrop(384, 384),
    alb.HorizontalFlip(p=0.5),
])

# https://github.com/pytorch/vision/blob/main/references/segmentation/presets.py的SegmentationPresetEval就是直接resize
test_transformer = alb.Compose([
    # alb.SmallestMaxSize(max_size=520),
    # alb.RandomCrop(480, 480)
    alb.Resize(384, 384)
])

norm_transform = alb.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))

# mask_transform = torchvision.transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())

# train_dataset = torchvision.datasets.voc.VOCSegmentation('../data', transform=train_transformer,
#                                                          target_transform=mask_transform)
# val_dataset = torchvision.datasets.voc.VOCSegmentation('../data', image_set='val', transform=test_transformer)
train_dataset = MyVocDataset('../data', transform=train_transformer, norm_transform=norm_transform)
val_dataset = MyVocDataset('../data', image_set='val', transform=test_transformer, norm_transform=norm_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# model = torch.load("../data/segmentation/190_unet_res34.pt")
encoder = ResNet34(20, small_scale=False)
# decoder = UnetHead(21, activation='')
# model = Unet(encoder, decoder)

# model.load_state_dict(torch.load('../data/segmentation/190_unet_res34.pth'))

# model = torch.nn.DataParallel(model)
# decoder = FCNDecoder(512, 21)
# model = FCN(encoder, decoder)

# ------------------------
decoder = FPNDecoder(encoder.channels)
model = FPN(encoder, decoder, 21)

# model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, pretrained_backbone=False)
# model.classifier[4] = torch.nn.Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))
# model = model.train().to(device)

trainner = SegTrainner(train_dataset, val_dataset, model, 21, '../data/pacal', lr=0.0001)
trainner.batch_size = 8
trainner.train()
