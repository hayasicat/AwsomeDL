# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 11:26
# @Author  : ljq
# @desc    : 特殊的结构，之后尝试从这边剥离
# @File    : STN.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, model_func, input_chans=3):
        super().__init__()
        self.model = model_func(6, input_chans=input_chans)
        # 初始化矩阵的时候要调整一下初始参数，最好可以选择不同的lr衰减
        self.model.fc.weight.data.zero_()
        self.model.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.model(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class MonoDepthSTN(nn.Module):
    # 单目的STN
    """
    inputs 图片以及相机内参，输出是一/多张图片
    monodepth2 里面是一次性吃2帧进去,吐出两帧的姿态
    # TODO：1. 吃进去两帧吐出来两个姿态效果咋样
            2. 套一个比较复杂的pose_encoder和直接一个fc输出有什么区别
            3. depth_net与

    """

    def __init__(self, depth_net, pose_encoder):
        super(MonoDepthSTN, self).__init__()
        self.depth_net = depth_net
        # 6个输入，6个输出
        self.pose_net = pose_encoder(6, input_chans=6)
        # 一开始的初始化都给0
        self.pose_net.fc.weight.data.zero_()
        self.model.fc.bias.data.copy_(torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float))

        self.frame_idx = [-1, 0, 1]
        self.prime_name = 'prime'

    def get_pose(self, pre_image, cur_image, next_image):
        # 前后帧
        refer_input = torch.cat([pre_image, cur_image], dim=1)
        next_input = torch.cat([cur_image, next_image], dim=1)
        # refer才是一个逆变换,以向后变换来说
        refer_trans = self.pose_net(refer_input.to(torch.float32))
        next_trans = self.pose_net(next_input.to(torch.float32))
        return refer_trans, next_trans

    def depth_map(self, inputs):
        # TODO：输入的图片要归一化，这样子写的也不太灵活
        depth_maps = self.depth_net(inputs['prime0'].to(torch.float32))
        return depth_maps

    def forward(self, pre_image, cur_image, next_image):
        # 姿态估计，后面更改为边佳旺的前后两帧

        refers_trans, next_trans = self.get_pose(pre_image, cur_image, next_image)
        # 跑出一堆多尺度的深度图出来
        depth_maps = self.depth_map(cur_image)
        return depth_maps, refers_trans, next_trans
