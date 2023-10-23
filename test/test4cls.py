# _*_ coding:utf-8 _*_
# @author:ljq
# @date: 2023/8/8 22:46
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from Base.BackBone import ResNet34, ResNet18, Vgg16
from Base.Metrics.CLS import Accuracy

device = torch.device("cuda")

train_transformer = torchvision.transforms.Compose([
    # torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.3)),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # torchvision.transforms.RandomVerticalFlip(),
    # torchvision.transforms.RandomRotation(15),
    # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # torchvision.transforms.PILToTensor(),
    # torchvision.transforms.ConvertImageDtype(torch.float),

    # torchvision.transforms.RandomErasing(p=0.7),
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

train_dataset = torchvision.datasets.CIFAR100(root='../data', download=True, transform=train_transformer)
test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=test_transformer)
# for i in range(10):
#     img = train_dataset[i][0].permute(1, 2, 0)
#     print(img)
#     print(torch.max(img))
#     print(img.size)
#     plt.imshow(img)
#     plt.show()
train_loader = DataLoader(train_dataset, batch_size=900, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

model = ResNet34(100)
# model = resnet34git(100)
model.init_weights()
model = torchvision.models.resnet34(False)
# model.fc = nn.Linear(model.fc.in_features, 100)
# model = model.train().to(device)
# 多卡并行训练

model = nn.DataParallel(model)
model = model.train().to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[60, 120, 160], gamma=0.2)
# sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.01, epochs=100, steps_per_epoch=len(train_loader))
ce_loss = nn.CrossEntropyLoss()
epoch = 200

for e in range(epoch):
    model.train()
    total_loss = []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        opt.zero_grad()
        labels = labels.to(device)
        pred = model(imgs)
        train_loss = ce_loss(pred, labels)
        train_loss.backward()
        opt.step()
        total_loss.append(train_loss.detach().cpu().numpy())
    sched.step()
    print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / len(total_loss)))
    if e % 5 == 0:
        model.eval()
        with torch.no_grad():
            for img, labels in test_loader:
                img = img.to(device)
                pred = model(img)
                pred_cls = torch.argmax(pred, dim=1)

                Accuracy(pred, labels)
        print('epoch is {}, acc is {}'.format(e, Accuracy.acc()))
        Accuracy.clean()
