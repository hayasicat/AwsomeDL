# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 17:09
# @Author  : ljq
# @desc    : 
# @File    : trainner.py
import torch
from torch.utils.data import DataLoader


class SegTrainner:
    lr = 0.001
    weight_decay = 5e-4
    epochs = 200
    batch_size = 600
    num_workers = 4

    def __init__(self, train_dataset, test_dataset, model):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        # 如果gpu可用的话
        if torch.cuda.is_available():
            # 选择多卡
            self.is_parallel = False
            if torch.cuda.device_count() > 1:
                self.is_parallel = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
