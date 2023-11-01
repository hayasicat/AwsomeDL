# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 15:25
# @Author  : ljq
# @desc    : 
# @File    : Seg.py
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUMetric:

    def __init__(self, class_num, *args, **kwargs):
        # classs_num 一般是指类别+1
        self.class_num = class_num
        self.per_img = []

    def clean(self):
        self.per_img = []

    def cal_hist(self, preds, gts):
        # 都是argmax的索引
        preds = preds.flatten()
        gts = gts.flatten()
        mask = (gts >= 0) & (gts < self.class_num)
        hist = np.bincount(gts[mask].astype(np.int64) * self.class_num + preds[mask].astype(np.int64),
                           minlength=self.class_num ** 2).reshape(self.class_num, self.class_num)
        # 上面的hist矩阵
        self.per_img.append(hist)
        return hist

    def single_iou(self, preds, gts):
        hist = self.cal_hist(preds, gts)
        iou = np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist) + 1e-7)
        return np.mean(iou[1:])

    def batch_miou(self, preds, gts):
        # 切片田间
        hist = np.zeros((self.class_num, self.class_num))
        for i in range(preds.shape[0]):
            hist += self.cal_hist(preds[i, :, :], gts[i, :, :])
        iou = np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist) + 1e-7)
        return np.mean(iou[1:])

    def miou(self):
        total_hist = sum(self.per_img)
        iou = np.diag(total_hist) / (total_hist.sum(axis=0) + total_hist.sum(axis=1) - np.diag(total_hist) + 1e-7)
        return np.mean(iou[1:])
