# _*_ coding:utf-8 _*_
# @author:ljq
# @date: 2023/8/8 22:46
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Base.BackBone import ResNet34, ResNet18, Vgg16, SimpleCnn, EfficientNetV2S, MobileNetV2, EfficientModify
from Base.Metrics.CLS import Accuracy
from Base.Utils.model_parameter_tuning import reset_dropout_prob
from Tools.Augmentation.MixUp import MixUpAug
from Tools.Logger.my_logger import init_logger
from Base.Loss.FocalLoss import MyFocalLoss
from Base.Utils.LRScheduler import WarmUp
import timm

device = torch.device("cuda")


def init_cifar100_logger(model):
    model_name = model.__class__.__name__
    root_path = '../data/cifar100_log/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    init_logger(root_path + model_name, 'training')


train_transformer = torchvision.transforms.Compose([
    # torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.3)),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # torchvision.transforms.RandomVerticalFlip(p=0.5),
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
# batch_size比较小会比较好
train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

# model = ResNet34(100, small_scale=True)
# model = resnet34git(100)
# model = SimpleCnn(100)
# TODO: EfficientNet 出了什么问题
model = EfficientNetV2S(100)
# model = MobileNetV2(100)
# model = EfficientModify(100)
model.init_weights()
init_cifar100_logger(model)
logger = logging.getLogger('training')
logger.info("==================new_line===============")
# 看看Timm的实现有没有问题
# model = timm.create_model("efficientnetv2_rw_s", num_classes=100)
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = torchvision.models.resnet34(False)
# model.fc = nn.Linear(model.fc.in_features, 100)
# model = model.train().to(device)
# 多卡并行训练
# torchvision.models.MobileNetV3
model = nn.DataParallel(model)
model = model.train().to(device)
# 非常难训练
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# sched = torch.optim.lr_scheduler.MultiStepLR(
#     opt, milestones=[80, 160, 230, 180], gamma=0.5)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[66, 126, 166], gamma=0.2)

# opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)

# sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.01, epochs=100, steps_per_epoch=len(train_loader))
# ce_loss = nn.CrossEntropyLoss()

ce_loss = MyFocalLoss(gamma=0, from_logits=True, num_classes=100)

epoch = 210
step = 0
warm_step = 50
warm_sched = WarmUp(opt, warm_step)

# Mixup初始化和变换
mix_up_trans = MixUpAug(0.1, 0.1)

for e in range(epoch):
    # model = reset_dropout_prob(e, model)
    model.train()
    total_loss = []
    for imgs, labels in train_loader:
        if step <= warm_step:
            warm_sched.step()
            step += 1
        labels_one_hot = F.one_hot(labels.to(torch.int64), num_classes=100)
        # imgs, labels_one_hot = mix_up_trans.transform(imgs, labels_one_hot)
        # if e >= 30:
        #     # TODO: 这边应该是One-hot
        #     imgs, labels_one_hot = mix_up(imgs, labels_one_hot, alpha=0.1)
        imgs = imgs.to(device)
        opt.zero_grad()
        labels = labels.to(device)
        labels_one_hot = labels_one_hot.to(device)
        pred = model(imgs)
        train_loss = ce_loss(pred, labels_one_hot, is_one_hot=True)
        # train_loss = ce_loss(pred, labels)
        train_loss.backward()
        opt.step()
        total_loss.append(train_loss.detach().cpu().numpy())
    sched.step()

    logger.info('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / len(total_loss)))
    if e % 5 == 0:
        model.eval()
        val_loss = []
        with torch.no_grad():
            for img, labels in test_loader:
                labels = labels.to(device)
                labels_one_hot = F.one_hot(labels.to(torch.int64), num_classes=100)
                img = img.to(device)
                labels_one_hot = labels_one_hot.to(device)
                pred = model(img)
                pred_cls = torch.argmax(pred, dim=1)
                val_l = ce_loss(pred, labels_one_hot, is_one_hot=True)
                # val_l = ce_loss(pred, labels)
                val_loss.append(val_l.detach().cpu().numpy())

                Accuracy(pred, labels)
        logger.info('epoch is {}, acc is {}, loss is {}'.format(e, Accuracy.acc(), sum(val_loss) / len(val_loss)))
        Accuracy.clean()
