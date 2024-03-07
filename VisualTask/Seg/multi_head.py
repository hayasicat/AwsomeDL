# -*- coding: utf-8 -*-
# @Time    : 2023/11/14 15:08
# @Author  : ljq
# @desc    : 
# @File    : multi_head.py
import os
import logging

import torch
from .trainner import SegTrainner
# from segmentation_models_pytorch.losses.dice import DiceLoss
from Base.Loss.FocalLoss import MyFocalLoss
from Base.Loss.DiceLoss import MyDiceLoss
from Base.Metrics.KP import KPDis
from Tools.Logger.my_logger import init_logger


class SegMultiHead(SegTrainner):

    def __init__(self, train_dataset, test_dataset, model, class_num=1, save_path=None, resume_path=None, lr=0.001,
                 batch_size=12):
        super().__init__(train_dataset, test_dataset, model, class_num, save_path, resume_path, lr,
                         batch_size=batch_size)
        self.kp_metric = KPDis()
        # self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.seg_criterion = MyFocalLoss(2, from_logits=True, ignore_index=255)
        self.cls_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.dice_loss = MyDiceLoss()
        # self.dice_loss = DiceLoss('multiclass', smooth=1, from_logits=False)

        self.kp_criterion = MyFocalLoss(2)  # 平方似乎会比较好点，超参数我觉得应该热力图的部分占比应该高一些
        # 分类计算
        # parameter
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 固定lr衰减的策略
        # self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[40, 70, 90, 120, 140, 180],
        #                                                   gamma=0.4)
        # 预训练权重可以缩短训练的时间
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[25, 45, 60, 75, 85, 93],
                                                          gamma=0.4)
        init_logger('/root/project/AwsomeDL/data/logs/multihead.log')
        self.logger = logging.getLogger('train')
        self.logger.info(self.save_path)

        # 减少保存的权重数量
        self.last_miou = -1

    def loss(self, pred, ips):
        seg = ips['seg'].to(self.device)
        heatmap = ips['kp_heatmap'].to(self.device)
        seg_pred = pred[0]
        heatmap_pred = pred[1]

        # 加上了dice loss来约束一下模型方面，收敛的速度还可以。0.47 ->
        seg_loss = self.seg_criterion(seg_pred, seg)
        d_loss = self.dice_loss(seg_pred, seg)
        # 求取heatmap得loss,mean效果实在是太差了
        # kp_loss = torch.mean((heatmap_pred - heatmap) ** 2)
        # kp_loss = self.kp_criterion(heatmap_pred, heatmap)
        # 加一下分类的结果
        l = 0.4 * seg_loss + 0.4 * d_loss
        if len(pred) >= 3:
            cls_label = ips['cls'].to(self.device)
            cls_pred = pred[2]
            # 这边看看结果
            l += 0.2 * self.cls_criterion(cls_pred, cls_label)
        return l

    def train(self):
        if self.epochs % 100 == 0:
            self.epochs += 1
        for e in range(self.epochs):
            total_loss = self.train_step()
            self.logger.info('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / (len(total_loss) + 1e-7)))
            self.sched.step()
            if e % 10 == 0:
                text_echo, miou = self.val_step()
                self.logger.info('epoch is {}   '.format(e) + text_echo)
                self.save_checkpoints(e, miou)

    def train_step(self):
        self.model.train()
        total_loss = []
        for ips in self.train_loader:
            imgs = ips['img']
            imgs = imgs.to(self.device)
            self.optim.zero_grad()
            pred = self.model(imgs)
            # 255有效值不变，需要提取边缘得mask码来做
            train_loss = self.loss(pred, ips)
            train_loss.backward()
            self.optim.step()
            total_loss.append(train_loss.detach().cpu().numpy())
        return total_loss

    @torch.no_grad()
    def val_step(self):
        self.model.eval()
        for ips in self.train_loader:
            imgs = ips['img']
            seg = ips['seg']
            heatmap = ips['kp_heatmap']

            imgs = imgs.to(self.device)
            seg = seg.to(self.device)
            heatmap = heatmap.to(self.device)
            preds = self.model(imgs)
            seg_preds = preds[0]
            seg_preds = torch.softmax(seg_preds, dim=1)
            seg_preds = torch.argmax(seg_preds, dim=1)
            seg_preds = seg_preds.cpu().numpy()

            kp_preds = preds[1]
            self.kp_metric.batch_heatmap_dis(kp_preds, heatmap)
            gt = seg.cpu().numpy()
            # heat_map先不计算
            self.metric.batch_miou(seg_preds, gt)

        # 计算所有的miou
        miou = self.metric.miou()
        self.metric.clean()
        distance = self.kp_metric.distances()
        text_echo = 'miou: {},distance:{}'.format(miou, distance)
        self.kp_metric.clearn()
        return text_echo, miou

    def resume_from(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_checkpoints(self, e, miou=-1):
        self.model.eval()
        if self.is_parallel:
            model = self.model.module
        else:
            model = self.model
        if miou > self.last_miou:
            # 保存最好的
            torch.save(model.state_dict(), os.path.join(self.save_path, 'best.pth'))
        # 保存最后一次的
        torch.save(model.state_dict(), os.path.join(self.save_path, 'last.pth'))
        self.model.train()