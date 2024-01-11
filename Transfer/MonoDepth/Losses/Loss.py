# -*- coding: utf-8 -*-
# @Time    : 2024/1/2 14:28
# @Author  : ljq
# @desc    : 
# @File    : Loss.py
import torch
from torch.nn.modules.loss import _Loss

from ._depth_loss_utils import SSIM


class ReprojectLoss(_Loss):
    """
    重投影误差
    """

    def __init__(self, has_ssim=True):
        super().__init__()
        self.has_ssim = has_ssim
        if has_ssim:
            self.ssim = SSIM()

    def forward(self, pred, target):
        abs_diff = torch.abs(pred - target)
        l1_loss = abs_diff.mean(1, True)
        if not self.has_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss


class EdgeSmoothLoss(_Loss):
    def __init__(self, need_norm=True):
        super().__init__()
        self.need_norm = need_norm

    def forward(self, disp, img):
        if self.need_norm:
            # 正则化
            mean_disp = disp.mean(2, True).mean(3, True)
            disp = disp / (mean_disp + 1e-7)
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()


class AutoMask(_Loss):
    def __init__(self, use_auto_mask=True, use_multi_scale=False, device=torch.device("cpu")):
        super().__init__()
        self.use_auto_mask = use_auto_mask
        self.use_multi_scale = use_multi_scale
        self.loss_backend = ReprojectLoss()
        self.device = device

    def compute_real_project(self, cur_images, pre_images, next_images):
        """
        计算auto mask 需要用到的东西
        :param cur_images:
        :param pre_images:
        :param next_images:
        :return:
        """
        identity_reprojection_losses = []
        identity_reprojection_losses.append(self.loss_backend(pre_images, cur_images))
        identity_reprojection_losses.append(self.loss_backend(next_images, cur_images))
        mask = torch.cat(identity_reprojection_losses, 1)
        mask += torch.randn(mask.shape, device=self.device) * 0.00001
        return mask

    def forward(self, pre_pred, cur_image, mask):
        # 用不同的loss来约束
        reprojection_loss = self.loss_backend(pre_pred, cur_image)
        combined = torch.cat((mask, reprojection_loss), dim=1)
        losses, idx = torch.min(combined, dim=1)
        return losses
