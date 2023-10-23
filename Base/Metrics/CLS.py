# -*- coding: utf-8 -*-
# @Time    : 2023/10/11 16:21
# @Author  : ljq
# @desc    : 
# @File    : CLS.py
import threading

import torch


class Accuracy:
    _instance = None
    _correct_samples = 0
    _total_samples = 0

    def __init__(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        with threading.Lock():
            # 线程锁，防止不同线程同时初始化
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def clean(cls):
        cls._correct_samples = 0
        cls._total_samples = 0

    @classmethod
    def accumulate(cls, pred, labels):
        # 全部叠到CPU上面执行，如果pred最后一个维度不是1的话那么对它的dim进行降维。
        pred = pred.cpu()
        labels = labels.cpu()
        # 维度判断
        if len(pred.size()) >= 2 and pred.size(-1) > 1:
            # dim降维
            pred = torch.argmax(pred, dim=1)
        # 判断对错
        result = pred == labels
        # 统计正确与否
        cls._correct_samples += result.sum().numpy()
        cls._total_samples += result.size(0)

    @classmethod
    def acc(cls):
        return cls._correct_samples / float(cls._total_samples)

    def __call__(self, *args, **kwargs):
        self.accumulate(*args, **kwargs)
