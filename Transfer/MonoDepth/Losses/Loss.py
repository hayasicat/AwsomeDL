# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 14:28
# @Author  : ljq
# @desc    : 
# @File    : Loss.py
import torch
from torch.nn.modules.loss import _Loss

from ._depth_loss_utils import SSIM


class ReprojectLoss(_Loss):
    def __init__(self, has_ssim=False):
        super().__init__()
        self.has_ssim = has_ssim
        if has_ssim:
            self.ssim = SSIM()

    def __forward__(self, pred, target):
        abs_diff = torch.abs(pred - target)
        l1_loss = abs_diff.mean(1, True)
        if not self.has_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss
