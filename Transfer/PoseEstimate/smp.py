# -*- coding: utf-8 -*-
# @Time    : 2024/3/5 15:56
# @Author  : ljq
# @desc    : 这是用segment model pytorch来做分割
# @File    : smp.py

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
import torch
import torch.nn as nn
from VisualTask.Seg.trainner import SegTrainner
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18
from Transfer.VisualFLS.dataset import FLSDataset, FLS_norm_transform, FLS_test_transforms, FLS_train_transforms

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import albumentations as albu

device = torch.device('cuda')

train_dataset = FLSDataset('/root/data/VisualFLS', is_crop=False, transforms=FLS_train_transforms,
                           norm_transforms=FLS_norm_transform)
val_dataset = FLSDataset('/root/data/VisualFLS', is_crop=False, is_train=False, transforms=FLS_test_transforms,
                         norm_transforms=FLS_norm_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
test_loader = DataLoader(val_dataset, batch_size=10, shuffle=True,
                         num_workers=4)

model = smp.PAN(
    encoder_name='resnet18',
    encoder_weights='imagenet',
    classes=3)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = smp.utils.losses.CrossEntropyLoss()
        self.dice = smp.utils.losses.DiceLoss()

    def forward(self, input, target):
        print(input.size(), target.size())
        return 0.5 * self.ce(input, target) + 0.5 * self.dice(input, target)


loss = smp.utils.losses.CrossEntropyLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5, activation='argmax2d'),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)

max_score = 0
save_root = r'/root/project/AwsomeDL/data/lockhole/smp'
for i in range(0, 40):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(test_loader)
    print(valid_logs['iou_score'])
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, os.path.join('best_model.pth'))
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
