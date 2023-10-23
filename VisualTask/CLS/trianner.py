# _*_ coding:utf-8 _*_
# @author:ljq
# @date: 2023/8/8 22:46
# @desc: 可以注册进hook用来保存

import torch
from torch.utils.data import DataLoader
from torch import nn

from Base.BackBone import ResNet34
from Base.Metrics.CLS import Accuracy


class CLSTrainer:
    lr = 0.001
    weight_decay = 5e-4
    epochs = 200
    batch_size = 600
    num_workers = 4

    def __init__(self, train_dataset, test_dataset, model):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        # 如果gpu可用的话
        if torch.cuda.is_available():
            # 选择多卡
            self.is_parallel = False
            if torch.cuda.device_count() > 1:
                self.is_parallel = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[60, 120, 160], gamma=0.2)
        ce_loss = nn.CrossEntropyLoss()
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.train().to(self.device)
        for e in range(self.epochs):
            self.model.train()
            total_loss = []
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(imgs)
                train_loss = ce_loss(pred, labels)
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                total_loss.append(train_loss.detach().cpu().numpy())
            sched.step()
            print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / len(total_loss)))
            if e % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    for img, labels in test_loader:
                        img = img.to(self.device)
                        pred = self.model(img)
                        Accuracy(pred, labels)
                print('epoch is {}, acc is {}'.format(e, Accuracy.acc()))
                Accuracy.clean()
