# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 17:08
# @Author  : ljq
# @desc    : 训练检测一个锁孔
# @File    : test4lockedhole.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import albumentations as alb
from PIL import Image

from VisualTask.Seg.trainner import SegTrainner
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18
from Transfer.VisualFLS.dataset import FLSDataset, FLS_norm_transform, FLS_test_transforms, FLS_train_transforms

device = torch.device('cuda')

train_dataset = FLSDataset('/backup/VisualFLS', transforms=FLS_train_transforms, norm_transforms=FLS_norm_transform)
val_dataset = FLSDataset('/backup/VisualFLS', is_train=False, transforms=FLS_test_transforms,
                         norm_transforms=FLS_norm_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

encoder = ResNet34(20, small_scale=False)
decoder = UnetHead(2, activation='')
model = Unet(encoder, decoder)

# model.load_state_dict(torch.load('../data/lockhole/190_unet_res34.pth'))

model = torch.nn.DataParallel(model)
model = model.train().to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[60, 120, 160], gamma=0.2)
ce_loss = nn.CrossEntropyLoss()
epoch = 200
root_path = '../data/lockhole'
if not os.path.exists(root_path):
    os.makedirs(root_path)
for e in range(epoch):
    model.train()
    total_loss = []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        opt.zero_grad()
        labels = labels.type(torch.LongTensor).to(device)
        pred = model(imgs)
        # 255有效值不变，需要提取边缘得mask码来做
        train_loss = ce_loss(pred, labels)
        train_loss.backward()
        opt.step()
        total_loss.append(train_loss.detach().cpu().numpy())
    sched.step()
    print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / (len(total_loss) + 1e-7)))
    if e % 10 == 0:
        model.eval()
        save_name = "{}_unet_res34.pth".format(e)
        torch.save(model.module.state_dict(), os.path.join(root_path, save_name))
        # for imgs, labels in test_loader:
        #     imgs = imgs.to(device)
        #     labels = labels.type(torch.LongTensor).to(device)
        #     preds = model(imgs)
        #     preds = torch.softmax(preds, dim=1)
        #     preds = torch.argmax(preds, dim=1)
        #     preds = preds.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
        #         (preds.shape[1], preds.shape[2]))
        #     gt = labels.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
        #         (labels.shape[1], labels.shape[2]))
        #     plt.imshow(preds)
        #     plt.show()
        #     plt.imshow(gt)
        #     plt.show()
