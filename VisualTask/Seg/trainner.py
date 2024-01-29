# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 17:09
# @Author  : ljq
# @desc    : 
# @File    : trainner.py
import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from Base.Metrics.SEG import IOUMetric
from Base.Loss.DiceLoss import MyDiceLoss
from Base.Loss.FocalLoss import MyFocalLoss


class SegTrainner:
    weight_decay = 5e-4
    epochs = 200
    batch_size = 24
    val_ratio = 2
    num_workers = 4

    def __init__(self, train_dataset, test_dataset, model, class_num=1, save_path=None, resume_path=None, lr=0.001,
                 **kwargs):
        self.lr = lr
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.save_path = save_path
        self.metric = IOUMetric(class_num)
        # TODO: 把后面这一块东西给移动到别的地方去
        if torch.cuda.is_available():
            # 选择多卡
            self.is_parallel = False
            if torch.cuda.device_count() > 1:
                self.is_parallel = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("is_parallel: {} , gpu_number:{}".format(self.is_parallel, torch.cuda.device_count()))
        if not resume_path is None:
            self.resume_from(resume_path)
        if self.is_parallel:
            self.train_loader, self.test_loader = self.DataParallel()
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size * self.val_ratio, shuffle=True,
                                          num_workers=4)
        self.model.to(self.device)
        # parameter
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 固定lr衰减的策略
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[50, 100, 140, 180], gamma=0.3)
        # 不太稳定需要更改loss
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        # self.criterion = MyFocalLoss(ignore_index=255, from_logits=True)
        self.dice_loss = MyDiceLoss(ignore_index=255)

        if not os.path.exists(self.save_path):
            os.makedirs(save_path)

    def DataParallel(self):
        self.model = torch.nn.DataParallel(self.model)
        trian_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size * self.val_ratio, shuffle=True,
                                 num_workers=4)
        return trian_loader, test_loader

    def Distribute(self):
        # 把所有的BN都给同步了
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
        # drop-last干嘛的
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=4,
                                  drop_last=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size * self.val_ratio, sampler=test_sampler,
                                 num_workers=4)
        return train_loader, test_loader

    def train(self):
        for e in range(self.epochs):
            total_loss = self.train_step()
            print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / (len(total_loss) + 1e-7)))
            self.sched.step()
            if e % 10 == 0:
                self.save_checkpoints(e)
                text_echo = self.val_step()
                print('epoch is {}, '.format(e) + '   ' + text_echo)

    def train_step(self):
        self.model.train()
        total_loss = []
        for imgs, labels in self.train_loader:
            imgs = imgs.to(self.device)
            self.optim.zero_grad()
            labels = labels.type(torch.LongTensor).to(self.device)
            pred = self.model(imgs)
            # 255有效值不变，需要提取边缘得mask码来做
            seg_loss = self.criterion(pred, labels)
            iou_loss = self.dice_loss(pred, labels)
            train_loss = 0.5 * seg_loss + 0.5 * iou_loss
            train_loss.backward()
            self.optim.step()
            total_loss.append(train_loss.detach().cpu().numpy())
        return total_loss

    @torch.no_grad()
    def val_step(self):
        self.model.eval()
        for imgs, labels in self.test_loader:
            imgs = imgs.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)
            preds = self.model(imgs)
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
            gt = labels.cpu().numpy()
            self.metric.batch_miou(preds, gt)
        # 计算所有的miou
        miou = self.metric.miou()
        self.metric.clean()
        text_echo = 'miou is {}'.format(miou)
        return text_echo

    def resume_from(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("resum from {}".format(model_path))

    def save_checkpoints(self, e):
        self.model.eval()
        save_name = "{}_model.pth".format(e)
        if self.is_parallel:
            model = self.model.module
        else:
            model = self.model
        torch.save(model.state_dict(), os.path.join(self.save_path, save_name))
