# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 11:43
# @Author  : ljq
# @desc    : 
# @File    : DiceLoss.py
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MyDiceLoss(_Loss):
    def __init__(self, from_logits=True, ignore_index=None):
        super().__init__()
        self.from_logits = from_logits
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, smooth=1.0, eps=1e-7):
        # 现在默认是多类别
        if self.from_logits:
            y_pred = torch.log_softmax(y_pred, dim=1).exp()
        b, c, h, w = y_pred.size()
        # 用来变换
        dims = (0, 2)
        y_target = y_target.view(b, -1)
        y_pred = y_pred.view(b, c, -1)
        if not self.ignore_index is None:
            mask = y_target != self.ignore_index
            # 为了测试直接用0作为掩码
            # mask = y_target > self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)
            y_target = F.one_hot(y_target * mask, c).permute(0, 2, 1).to(torch.long)
        else:
            y_target = F.one_hot(y_target, c).permute(0, 2, 1).to(torch.long)
        # 多分类计算
        y_target = y_target.type_as(y_pred)
        intersection = torch.sum(y_pred * y_target, dim=dims)
        cardinality = torch.sum(y_pred + y_target, dim=dims)
        # 如果不适用1.0的话会导致问腿
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        # 去除
        mask = y_target.sum(dims) > 0
        dice_loss *= mask.to(dice_loss.dtype)
        return dice_loss.mean()

# if __name__ == '__main__':
#     # 对比一下smp
#     p = torch.randint(1, 9, (3, 3, 3))
#     p = F.one_hot(p, 10).permute(0, 3, 1, 2)
#     t = torch.randint(1, 9, (3, 3, 3))
#     l = MyDiceLoss(10, ignore_index=5)
#     from segmentation_models_pytorch.losses.dice import DiceLoss
#
#     smp = DiceLoss('multiclass', smooth=1, from_logits=False)
#     print(p.size())
#     print(t)
#     print(l(p, t, smooth=1))
#     print(smp(p, t))
#     print(smp.classes, smp.ignore_index,smp.log_loss)
