# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 18:40
# @Author  : ljq
# @desc    : 
# @File    : KP.py
import torch
import numpy as np


class KPDis():
    def __init__(self):
        self.dis = np.empty(0)

    def batch_heatmap_dis(self, heatmaps, gt_heatmaps):
        b, c, h, w = heatmaps.size()
        for i in range(b):
            dis = self.heatmap_dis(heatmaps[i, :, :, :], gt_heatmaps[i, :, :, :])
            self.dis = np.hstack([self.dis, dis])

    def heatmap_dis(self, heatmaps, gt_heatmaps):
        c, h, w = heatmaps.size()
        heatmaps_view = heatmaps.view(c, -1)
        gt_view = gt_heatmaps.view(c, -1)
        pred_idx = torch.argmax(heatmaps_view, dim=1).cpu().detach().numpy()
        pred_col = np.floor(pred_idx / w)
        pred_row = pred_idx - pred_col * w
        pred_coor = np.hstack([pred_row.reshape(-1, 1), pred_col.reshape(-1, 1)]).astype(np.int32)

        # 计算一下gt的最大值位置
        gt_idx = torch.argmax(gt_view, dim=1).cpu().detach().numpy()
        gt_col = np.floor(gt_idx / w)
        gt_row = gt_idx - gt_col * w
        gt_coor = np.hstack([gt_row.reshape(-1, 1), gt_col.reshape(-1, 1)]).astype(np.int32)
        # 计算gt最大值是否超过0.5
        mask = (torch.max(gt_view, dim=1).values.cpu().numpy() > 0.2)
        pred_coor[0, 1] = pred_coor[0, 1] + 10
        distance = np.sqrt(np.sum(np.square(gt_coor[mask, :] - pred_coor[mask, :]), axis=1))
        return distance

    def distances(self):
        return self.dis.mean()

    def clearn(self):
        self.dis = np.empty(0)
