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


class PoseDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        input_channel = self.encoder.channels[-1]
        self.decoder = nn.Sequential(*[
            nn.Conv2d(input_channel, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6, 1),
        ])

    def forward(self, input_tensor):
        features = self.encoder.feature_extract(input_tensor)
        last_feature = features[-1]
        out = self.decoder(last_feature)
        out = out.mean(3).mean(2)
        return out


class ClippedRelu(nn.Module):
    def __init__(self):
        super(ClippedRelu, self).__init__()

    def forward(self, x):
        return torch.clamp(x, min=-1.0, max=1.0)


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
        # TODO: 对齐一下monodepth pose decoder
        encoder = pose_encoder(6, input_chans=6)
        # self.pose_net = PoseDecoder(encoder)
        # 一开始的初始化都给0
        self.pose_net = encoder
        # self.pose_net.fc.weight.data.zero_()
        # self.pose_net.fc.bias.data.copy_(torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001], dtype=torch.float))
        # self.pose_net.fc.bias.data.copy_(torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.001], dtype=torch.float))

        self.frame_idx = [-1, 0, 1]
        self.prime_name = 'prime'
        # self.clamp_relu = ClippedRelu()

    def get_pose(self, pre_image, cur_image):
        # 前后帧
        refer_input = torch.cat([pre_image, cur_image], dim=1)
        # refer才是一个逆变换,以向后变换来说
        refer_trans = self.pose_net(refer_input)
        # 对齐一下monodepth2
        # refer_trans = self.pose_net(refer_input)
        # 因为后面是 nxn的matrix所以需要reshape一下
        # @ljq: Z轴和R角值太大导致训不起来，乘以一个固定的系数
        # 系数是一个超参的值，调越大训练可能越不稳定
        # 系数是一个超参数的值，越大的话更容易控制姿态网络的一个跳变幅度，这样子的结果更容易保证一个姿态输出的稳定性
        refer_trans = 0.1 * refer_trans.view(pre_image.size(0), -1, 6)
        refer_trans[..., -1] = 0.05 * refer_trans[..., -1]
        refer_trans[..., :3] = 0.01 * refer_trans[..., :3]
        # 防止超出
        # refer_trans[..., 3:] = torch.clamp(refer_trans[..., 3:], min=-1.0, max=1.0)
        # 给一个物理意义
        return refer_trans

    def depth_map(self, cur_images):
        # TODO：输入的图片要归一化，这样子写的也不太灵活
        depth_maps = self.depth_net(cur_images)
        return depth_maps[::-1]

    def forward(self, pre_image, cur_image, next_image, *args, **kwargs):
        # 姿态估计，后面更改为边佳旺的前后两帧
        prex_x = (pre_image - 0.45) / 0.225
        cur_x = (cur_image - 0.45) / 0.225
        next_x = (next_image - 0.45) / 0.225

        refers_pose = self.get_pose(prex_x, cur_x)
        next_pose = self.get_pose(cur_x, next_x)
        # 跑出一堆多尺度的深度图出来
        depth_maps = self.depth_map(cur_x)
        # 从大大小深度图
        return depth_maps, refers_pose, next_pose


class MonoDepthPair(MonoDepthSTN):

    def __init__(self, depth_net, pose_encoder):
        super(MonoDepthPair, self).__init__(depth_net, pose_encoder)

    def forward(self, cur_image, refers_images, *args, **kwargs):
        cur_x = (cur_image - 0.45) / 0.225
        refers_x = [(img - 0.45) / 0.225 for img in refers_images]

        pose = []
        pose_inv = []
        refers_depth_maps = []
        cur_depth_map = self.depth_map(cur_x)

        # 正向姿态和逆向姿态
        for x in refers_x:
            pose.append(self.get_pose(cur_x, x))
            pose_inv.append(self.get_pose(x, cur_x))
            # 深度图
            refers_depth_maps.append(self.depth_net(x))

        return cur_depth_map, refers_depth_maps, pose, pose_inv
