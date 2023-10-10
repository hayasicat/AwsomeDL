# _*_ coding:utf-8 _*_
# @author:ljq
# @date: 2023/8/8 22:46
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn

from Base.BackBone import ResNet34

device = torch.device("cuda")

train_transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_transformer = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

train_dataset = torchvision.datasets.CIFAR100(root='./data', download=True, transform=train_transformer)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transformer)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = ResNet34(100, 32)
model.init_weights()
model = model.train().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()
epoch = 20

for e in range(epoch):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(len(train_loader))
    total_loss = []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred = model(imgs)
        opt.zero_grad()
        train_loss = ce_loss(pred, labels)
        train_loss.mean().backward()
        opt.step()
        total_loss.append(train_loss.detach().cpu().numpy())
    print(sum(total_loss))

if __name__ == '__main__':
    pass
