# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 17:00
# @Author  : ljq
# @desc    : 
# @File    : trainner.py
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from Transfer.MonoDepth.MonoUtils.CameraTrans import transformation_from_parameters, get_sample_grid
from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask
from Transfer.MonoDepth.MonoUtils import disp_to_depth


# 如果最后一层是torch.sigmoid的话，不如给一个nn.relu6合适，然后缩放nn.relu6到0-1
class MonoTrainer():
    def __init__(self):
        pass


def view_syn(source_frame, grid_sample):
    # 视角转换的loss
    source_frame = source_frame.to(grid_sample.dtype)
    new_view = F.grid_sample(
        source_frame,
        grid_sample,
        padding_mode="zeros")
    return new_view


def visualTensor(image_tensor):
    # batchsize就一第一张
    if len(image_tensor.size()) > 3:
        image_tensor = image_tensor[0, ...]
    new_image = image_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    plt.imshow(new_image)
    plt.show()


def viewDepth(depth):
    if len(depth.size()) > 3:
        depth = depth[0, ...]
    depth = depth.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    plt.imshow(depth)
    plt.show()


def visualReproject(loss_tensor):
    # 重投影的loss可视化
    if len(loss_tensor.size()) > 3:
        loss_tensor = loss_tensor[0, ...]
    loss_tensor = loss_tensor.detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(loss_tensor)
    plt.show()


max_depth = 40
min_depth = 1

if __name__ == "__main__":
    from Transfer.MonoDepth.dataset import MonoDataset

    data_root = r'/root/project/AwsomeDL/data/BowlingMono'
    train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
    train_data = MonoDataset(data_root, train_file_path, 832, 1824)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_data, 1)
    # 模型初始化
    from Base.BackBone.STN import MonoDepthSTN
    from Base.BackBone.EfficientNetV2 import EfficientNetV2S
    from Base.BackBone.ResNet import ResNet18, ResNet34
    from Base.SegHead.DepthHead import DepthDecoder, DepthNet

    # 初始化以后
    # encoder = EfficientNetV2S(10, input_chans=3)
    encoder = ResNet34(10, input_chans=3)

    depth_decoder = DepthDecoder(encoder.channels)
    depth_net = DepthNet(encoder, depth_decoder)
    # TODO: monodepth2预测出来的是两张图片两张深度，但是我这边就直接两张图片一个姿态
    model = MonoDepthSTN(depth_net, ResNet18)
    # 深度网络的深度肯定要比姿态估计的深度要低一些
    opt = torch.optim.Adam([{'params': model.depth_net.parameters()},
                            {'params': model.pose_net.parameters(), 'lr': 1e-4}], lr=1e-4, weight_decay=5e-4)
    # opt = torch.optim.Adam([{'params': model.depth_net.parameters()}], lr=1e-3, weight_decay=5e-4)
    # sched = torch.optim.lr_scheduler.MultiStepLR(
    #     opt, milestones=[80, 160, 230, 180], gamma=0.5)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[66, 126, 166], gamma=0.2)

    g_loss = ReprojectLoss()
    s_loss = EdgeSmoothLoss()
    s_weight = 1e-3
    auto_mask = AutoMask()

    for idx, inputs in enumerate(train_loader):
        if idx < 12:
            continue
        # monodepth2的训练方式
        # TODO: 输入图片一开始直接给个归一化来大力试试，不然loss爆炸
        cur_image = inputs['prime0_0']
        pre_image = inputs['prime-1_0']
        next_image = inputs['prime1_0']
        K = inputs['K_0']
        inv_K = inputs['inv_K0']
        # 暂时使用一个scale
        # 计算出投影的矩阵
        # TODO： 因为automask会去找多个图片的重投影最小的区域，
        #  1. 所以如果图中有比较多的黑色区域他们会倾向于填充该区域.
        #  2. 如果图中弱纹理的区域越大的话，mask的区域也会越大，本来他是为了解决occusion的问题反倒会成为姿态估计的干扰
        #  3. 如果不适用这个的话，那么物体和物体间的边缘偏差都会比较高
        mask = auto_mask.compute_real_project(cur_image, pre_image, next_image)

        for i in range(20):
            depth_maps, refers_pose, next_pose = model(pre_image, cur_image, next_image)
            # 这里做一个scale up
            total_loss = 0
            for idx, depth_map in enumerate(depth_maps):
                depth_map = F.interpolate(depth_map, [cur_image.size(2), cur_image.size(3)], mode="bilinear",
                                          align_corners=False)
                # 强制最后一个维度为0试试
                # refers_pose[..., 4:] = 0
                # next_pose[..., 4:] = 0
                # print(refers_pose, next_pose)
                # TODO： 训练不起来，是不是要提高一下帧间差距
                # TODO： 既然训不起来，我觉得很大一个程度是因为分辨率的问题，小分辨率容易收敛，大分辨率很崩。跟我微调的分辨率应该是一个样子吧
                refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
                next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
                # 1. 如果固定好姿态输出的话应该是可以的,姿态估计的网络是比较难训练的
                _, depth = disp_to_depth(depth_map, min_depth, max_depth)
                # depth = depth_map
                # @ljq: 强制固定画面，看能不能学到姿态估计网络的样子
                # depth = torch.ones(depth_map.size())
                print(next_pose[..., 3:], refers_pose[..., 3:], depth.mean())

                refers_grid = get_sample_grid(depth, K, inv_K, refers_trans)
                next_grid = get_sample_grid(depth, K, inv_K, next_trans)

                pre2cur = view_syn(pre_image, refers_grid)
                next2cur = view_syn(next_image, next_grid)
                # visualTensor(pre2cur)
                # BBQ，
                pre_losses = auto_mask(pre2cur, cur_image, mask)
                next_losses = auto_mask(next2cur, cur_image, mask)
                current_loss = pre_losses.mean() + next_losses.mean() + s_weight * s_loss(depth_maps[-1], cur_image)
                total_loss += current_loss
                # 细节纹理来说，如果upsample的话很容易会遇到问题的，
                # TODO： 那么如果针对缩放度比较高的图片细节纹理会不会丢失很多
                print("第{}层输出,loss为{}".format(idx, current_loss))
                # total_loss += (g_loss(pre2cur, cur_image) + g_loss(next2cur, cur_image)).mean() + s_weight * s_loss(
                #     depth_maps[-1], cur_image)
                # 因为Z轴没有动
                # loss = (g_loss(next2cur, cur_image)).mean() + s_weight * s_loss(depth_maps[-1], cur_image)
            print(total_loss)
            total_loss.backward()
            opt.step()
        viewDepth(depth)
        visualTensor(next2cur * 255)
        visualTensor(cur_image * 255)
        visualTensor(next_image * 255)
        next_loss = auto_mask(next2cur, cur_image, mask)
        visualReproject(next_loss)

        next_loss = g_loss(next2cur, cur_image)
        visualReproject(next_loss)

        next_loss = g_loss(next_image, cur_image)
        visualReproject(next_loss)

        # 视角合成
        print(next_pose, next_trans)
        print(refers_pose.size(), next_pose.size())
        # 转换参数
        break

# TODO:
#  1. 姿态估计网络参数发生变动，深度网络参数发生变动导致训练的不稳定性。
#  2. 跨物体边缘差异比较大，这应该也不是什么问题。物体块里面的差异比较小，导致一些low，texture会有点问题
#  3. 去掉auto_mask是不可行的，因为这样子会导致一些self.egomotion的物体不能被去除
