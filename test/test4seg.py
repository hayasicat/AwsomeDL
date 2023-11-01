# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 9:13
# @Author  : ljq
# @desc    : 
# @File    : test4seg.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from Base.SegHead import FCNDecoder, FCN
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

train_transformer = alb.Compose([
    alb.SmallestMaxSize(max_size=224),
    alb.RandomCrop(224, 224),
    alb.HorizontalFlip(p=0.5),
])
test_transformer = alb.Compose([
    alb.SmallestMaxSize(max_size=224),
    alb.RandomCrop(224, 224)
])

norm_transform = alb.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                               std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

# mask_transform = torchvision.transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())

# train_dataset = torchvision.datasets.voc.VOCSegmentation('../data', transform=train_transformer,
#                                                          target_transform=mask_transform)
# val_dataset = torchvision.datasets.voc.VOCSegmentation('../data', image_set='val', transform=test_transformer)
train_dataset = MyVocDataset('../data', transform=train_transformer, norm_transform=norm_transform)
val_dataset = MyVocDataset('../data', image_set='val', transform=test_transformer, norm_transform=norm_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# model = torch.load("../data/segmentation/190_unet_res34.pt")
encoder = ResNet34(20, small_scale=False)
decoder = UnetHead(21, activation='')
model = Unet(encoder, decoder)

model.load_state_dict(torch.load('../data/segmentation/190_unet_res34.pth'))

# model = torch.nn.DataParallel(model)
# decoder = FCNDecoder(512, 21)
# model = FCN(encoder, decoder)

# model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, pretrained_backbone=False)
# model.classifier[4] = torch.nn.Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))

model = model.train().to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[60, 120, 160], gamma=0.2)
# sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.01, epochs=100, steps_per_epoch=len(train_loader))
ce_loss = nn.CrossEntropyLoss()
epoch = 200

root_path = '../data/segmentation'
if not os.path.exists(root_path):
    os.makedirs(root_path)
# # 载入模型
for e in range(epoch):
    model.train()
    total_loss = []
    for imgs, labels in train_loader:
        break
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
    # print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / len(total_loss)))
    if e % 10 == 0:
        model.eval()
        # save_name = "{}_unet_res34.pth".format(e)
        # torch.save(model.module.state_dict(),  os.path.join(root_path, save_name))
        # torch.save(model.cpu(), os.path.join(root_path, save_name))
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            preds = model(imgs)
            # print(preds.size())
            # preds = torch.softmax(preds, dim=1)
            # preds = torch.argmax(preds, dim=1)
            # preds = preds.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
            #     (preds.shape[1], preds.shape[2]))
            # gt = labels.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
            #     (labels.shape[1], labels.shape[2]))
            # plt.imshow(preds)
            # plt.show()
            # plt.imshow(gt)
            # plt.show()
            # break
            # print(preds)
