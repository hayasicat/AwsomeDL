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


class SegTrainner:
    lr = 0.001
    weight_decay = 5e-4
    epochs = 200
    batch_size = 8
    num_workers = 4

    def __init__(self, train_dataset, test_dataset, model, class_num=1, save_path=None, resume_path=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.save_path = save_path
        self.metric = IOUMetric(class_num)
        # 如果gpu可用的话
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
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        # parameter
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 固定lr衰减的策略
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[60, 120, 160], gamma=0.2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        for e in range(self.epochs):
            total_loss = self.train_step()
            print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / (len(total_loss) + 1e-7)))
            if e % 10 == 0:
                self.save_checkpoints(e)
                miou = self.val_step()
                print('epoch is {}, miou is {}'.format(e, miou))

    def train_step(self):
        self.model.train()
        total_loss = []
        for imgs, labels in self.train_loader:
            imgs = imgs.to(self.device)
            self.optim.zero_grad()
            labels = labels.type(torch.LongTensor).to(self.device)
            pred = self.model(imgs)
            # 255有效值不变，需要提取边缘得mask码来做
            train_loss = self.criterion(pred, labels)
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
        return miou

    def resume_from(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_checkpoints(self, e):
        self.model.eval()
        save_name = "{}_model.pth".format(e)
        torch.save(self.model.module.state_dict(), os.path.join(self.save_path, save_name))
