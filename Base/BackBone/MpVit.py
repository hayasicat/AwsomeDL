# -*- coding: utf-8 -*-
# @Time    : 2023/11/20 11:53
# @Author  : ljq
# @desc    : 
# @File    : MpVit.py
import torch
import torch.nn as nn
from ._utils import DPWConv, Conv3x3, PWDPWConv


class PatchEmbedStage(nn.Module):
    def __init__(self, embed_dim, num_path, stride=1):
        super(PatchEmbedStage, self).__init__()
        # Stride==2的时候就是用来缩小尺寸的，
        # 其实就是两个部分做的，MHA用来降采样，这个用来减少尺度大小
        # MPvit实现中的DepthWise中是没有bn和act层的
        self.PatchEmbeds = nn.ModuleList([
            DPWConv(embed_dim, embed_dim, stride if stride == 2 and i == 0 else 1, 1) for i in range(num_path)
        ])


class ResConv(PWDPWConv):
    def __init__(self, input_channel, output_channel):
        super().__init__(input_channel, output_channel)

    def forward(self, input_tensor):
        return self.model(input_tensor) + input_tensor


class MHCAEncoder(nn.Module):
    def __init__(self):
        super(MHCAEncoder, self).__init__()
        


class MHCAStage(nn.Module):
    def __init__(self, embed_dim, out_embed_dim, num_layer=1, num_heads=8, mlp_ratio=3, num_paths=4, drop_path_list=[]):
        super().__init__()
        self.conv_block = ResConv(embed_dim, embed_dim)
        # transformer的模块


class MpVitModel(nn.Module):
    def __init__(self, num_cls, input_chans=3, channels=[64, 128, 256, 512], num_paths=[2, 3, 3, 3], **kwargs):
        super().__init__()
        self.channels = channels
        self.num_stages = 4
        self.channels.insert(self.channels[0] // 2, 0)
        self.Conv1 = Conv3x3(input_chans, self.channels[0], stride=2, padding=1, act=nn.ReLU)
        self.Conv2 = Conv3x3(self.channels[0], self.channels[1], stride=2, padding=1, act=nn.ReLU)
        # 做Patch Embedding
        self.patch_embed_stage = nn.ModuleList([
            PatchEmbedStage(embed_dim=self.channels[idx + 1], num_path=num_paths[idx], stride=1 if idx == 0 else 0) \
            for idx in range(self.num_stages)
        ])

    pass
