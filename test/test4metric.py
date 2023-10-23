# -*- coding: utf-8 -*-
# @Time    : 2023/10/16 9:59
# @Author  : ljq
# @desc    : 
# @File    : test4metric.py
import torch

from Base.Metrics.CLS import Accuracy

if __name__ == "__main__":
    test_matric = 'acc'
    if test_matric == "acc":
        for i in range(1, 20):
            pred = torch.randint(i, (100, 1))
            gt = torch.randint(i, (100, 1))
            Accuracy(pred, gt)
            print(Accuracy.acc())
