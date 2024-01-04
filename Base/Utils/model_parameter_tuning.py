# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 11:42
# @Author  : ljq
# @desc    : 
# @File    : model_parameter_tuning.py
import torch
import torch.nn as nn


def reset_dropout_prob(epoch, model):
    p = None
    if epoch == 0:
        p = 0.1

    # if epoch == 0:
    #     p = 0.05
    # if epoch == 56:
    #     p = 0.1
    # if epoch == 141:
    #     p = 0.1

    if p is None:
        return model

    for n, layer in model.named_modules():
        if isinstance(layer, nn.Dropout):
            layer.p = p

    return model
