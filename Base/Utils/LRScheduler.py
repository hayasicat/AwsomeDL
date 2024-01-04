# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 15:19
# @Author  : ljq
# @desc    : 
# @File    : LRScheduler.py
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmUp(_LRScheduler):
    def __init__(self, optimizer, total_steps, gamma=0.1, last_epoch=-1, verbose=False):
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.total_steps = total_steps
        self.gamma = gamma
        super(WarmUp, self).__init__(optimizer, last_epoch, verbose)
    def get_lr(self) -> float:

        return [lr * min([self._step_count / self.total_steps, 1.0])
                for lr in self.base_lr]


if __name__ == "__main__":
    import torchvision

    opt = torch.optim.Adam(torchvision.models.mobilenet_v3_small(pretrained=False).parameters(), lr=0.001,
                           weight_decay=1e-3)
    s = WarmUp(opt, 200)
    for i in range(2000):
        print(opt.param_groups[0]['lr'])
        s.step()
    print(opt.param_groups[0]['lr'])
