# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 9:13
# @Author  : ljq
# @desc    : 
# @File    : test4seg.py
import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from albumentations import HorizontalFlip


from VisualTask.Seg.trainner import SegTrainner
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34

device = torch.device('cuda:0')

train_transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.RandomCrop(224, padding=4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),

    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),

]
)
mask_transform = torchvision.transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())

train_dataset = torchvision.datasets.voc.VOCSegmentation('../data', transform=train_transformer,
                                                         target_transform=mask_transform)
val_dataset = torchvision.datasets.voc.VOCSegmentation('../data', image_set='val', transform=test_transformer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

encoder = ResNet34(20, small_scale=True)
decoder = UnetHead(21)
model = Unet(encoder, decoder)
print(model(torch.ones((1, 3, 224, 224))).size())
model = model.train().to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[60, 120, 160], gamma=0.2)
# sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.01, epochs=100, steps_per_epoch=len(train_loader))
ce_loss = nn.CrossEntropyLoss()
epoch = 200

for i in range(10):
    # 测试一下alb图像增强的包
    images, target = train_dataset[i]
    print(type(images), type(target), images.size(), target.size())
    # mask 中有255表示的是空白区域或者是难以分割的物体
    # print(np.array(target).shape,np.max(np.array(target)))
    # print(np.unique(np.array(target)))
    # plt.imshow(target)
    # plt.show()
# # 载入模型
# for e in range(epoch):
#     model.train()
#     total_loss = []
#     for imgs, labels in train_loader:
#         imgs = imgs.to(device)
#         opt.zero_grad()
#         labels = labels.to(device)
#         pred = model(imgs)
#         train_loss = ce_loss(pred, labels)
#         train_loss.backward()
#         opt.step()
#         total_loss.append(train_loss.detach().cpu().numpy())
#     sched.step()
#     print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / len(total_loss)))
