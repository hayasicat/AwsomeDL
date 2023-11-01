# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 15:25
# @Author  : ljq
# @desc    : 
# @File    : Seg.py
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F


class IOU:
    _instance = None
    _per_img_iou = []

    def __init__(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        with threading.Lock():
            # 线程锁，防止不同线程同时初始化
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self, preds: torch.Tensor, gts, threshold=0.5, esp=1e-7, *args, **kwargs):
        # 如果preds的维度是三维，gts的维度也是三维,gts应该是要和preds同样的通道数
        iou = self.calculate_iou(preds, gts, threshold, esp)
        self._per_img_iou.append(iou)

    @classmethod
    def clean(self):
        self._per_img_iou = []

    @property
    def miou(self):
        # 单张图片IOU很容易被掩盖

        return sum([p['intersection'] for p in self._per_img_iou]) / (sum([p['union'] for p in self._per_img_iou]))

    @classmethod
    def iou(cls, preds: torch.Tensor, gts, threshold=0.5, esp=1e-7, *args, **kwargs):
        cls._per_img_iou.append(cls.calculate_iou(preds, gts, threshold, esp))

    @classmethod
    def get_iou(cls, preds: torch.Tensor, gts, threshold=0.5, esp=1e-7, *args, **kwargs):
        return cls.calculate_iou(preds, gts, threshold, esp)['iou']

    @staticmethod
    def calculate_iou(preds: torch.Tensor, gts, threshold=0.5, esp=1e-7, *args, **kwargs):
        # 手写一个iou ,具体的话，怎么算列表的gts。每个点的置信度也要确定,针对的是batch计算
        if not isinstance(gts, torch.Tensor):
            gts = torch.from_numpy(gts)
        # 检查维度，如果维度batch维度是1的话，则添加前面的维度。
        preds = preds.cpu()
        gts = gts.cpu()
        if len(preds.size()) < 4:
            preds = preds.unsqueeze(0)
        if len(gts.size()) < 3:
            gts = gts.unsqueeze(0)
        preds = torch.softmax(preds, dim=1)
        preds = (preds > threshold).type(preds.dtype)
        preds = preds.permute(0, 2, 3, 1)
        # 计算每个通道得IOU，把GT转为one-hot编码
        if preds.size()[1] >= 2:
            gts = F.one_hot(gts.long())
            # 将背景类别置为0
            gts[:, :, :, 0] = 0
            preds[:, :, :, 0] = 0
        # gts要转换为 C H W
        intersection = torch.sum(torch.mul(gts, preds), dim=(1, 2, 3))
        union = torch.sum(gts, dim=(1, 2, 3)) + torch.sum(preds, dim=(1, 2, 3)) - intersection + esp
        result = {'intersection': list(intersection.numpy()), 'union': list(union.numpy()),
                  'iou': (intersection + esp) / union}
        return result
