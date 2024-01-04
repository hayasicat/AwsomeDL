# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 18:37
# @Author  : ljq
# @desc    : 
# @File    : MixUp.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mix_up(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * images + (1 - lam) * images[index, :]
    mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    return mixed_x, mixed_labels


class MixUpAug():
    def __init__(self, p, alpha):
        self.p = max([0, min([p, 1])])
        self.alpha = alpha

    def transform(self, images, labels):
        # 根据百分比概率，进行mixup。
        labels = labels.to(images.dtype)
        mix_num = math.floor(images.size(0) * self.p)
        if self.p > 0 and mix_num >= 2:
            # 重复进来
            images[:mix_num, :], labels[:mix_num, :] = mix_up(images[:mix_num, :], labels[:mix_num, :], self.alpha)
        return images, labels


if __name__ == "__main__":
    i = torch.cat([torch.ones(4, 1, 3, 3), torch.zeros(3, 1, 3, 3)])
    l = F.one_hot(torch.cat([torch.ones(4), torch.zeros(3)]).to(torch.int64), num_classes=2).to(torch.float32)
    # mi, ml = mix_up(i, l, 0.2)
    t = MixUpAug(0.9, 0.1)
    i, l = t.transform(i, l)
    print(i)
    print(l)
