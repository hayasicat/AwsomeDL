# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 17:00
# @Author  : ljq
# @desc    : 
# @File    : trainner.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

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
        padding_mode="border")
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
    depth = depth.detach().cpu().permute(1, 2, 0).numpy()
    # 绘制归一化的深度图片，但是这种衡量很不准
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(depth.reshape((depth.shape[0], depth.shape[1])))[:, :, :3] * 255).astype(np.uint8)
    plt.imshow(colormapped_im)
    plt.show()


def visualReproject(loss_tensor):
    # 重投影的loss可视化
    if len(loss_tensor.size()) > 3:
        loss_tensor = loss_tensor[0, ...]
    loss_tensor = loss_tensor.detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(loss_tensor)
    plt.show()


# min_depth 设置的太大的话也不行，STN不愿意动
max_depth = 11
# 收敛以后T值的大小跟min_depth是有关系的
min_depth = 3


multi_scales = 4
s_weight = 1e-3


def VisualResult(model, input_info):
    auto_mask = AutoMask()
    cur_image_0 = input_info['prime0_0']
    pre_image_0 = input_info['prime-1_0']
    next_image_0 = input_info['prime1_0']
    K0 = input_info['K_0']
    inv_K0 = input_info['inv_K0']
    depth_maps, refers_pose, next_pose = model(pre_image_0, cur_image_0, next_image_0)

    refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
    next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
    mask = auto_mask.compute_real_project(cur_image_0, pre_image_0, next_image_0)
    _, depth = disp_to_depth(depth_maps[0], min_depth, max_depth)

    refers_grid = get_sample_grid(depth, K0, inv_K0, refers_trans)
    next_grid = get_sample_grid(depth, K0, inv_K0, next_trans)

    pre2cur = view_syn(pre_image_0, refers_grid)
    next2cur = view_syn(next_image_0, next_grid)

    # viewDepth(depth)
    visualTensor(next2cur * 255)
    visualTensor(cur_image_0 * 255)
    visualTensor(next_image_0 * 255)

    next_loss = auto_mask(next2cur, cur_image_0, mask)
    visualReproject(next_loss)

    next_loss = g_loss(next2cur, cur_image_0)
    visualReproject(next_loss)

    # next_loss = g_loss(next_image_0, cur_image_0)
    # visualReproject(next_loss)

    # 干脆把所有的深度图给可视化把，如果底层的深度图不会变成桶，那么可以用底层的深度图来收敛商城的深度图
    for dps in depth_maps:
        _, dp = disp_to_depth(dps, min_depth, max_depth)
        print(dp.mean())
        viewDepth(dp)
        time.sleep(0.1)
    print(next_pose, next_trans)
    print(refers_pose.size(), next_pose.size())


if __name__ == "__main__":
    from Transfer.MonoDepth.dataset import MonoDataset

    data_root = r'/root/project/AwsomeDL/data/BowlingMono'
    train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
    # train_data = MonoDataset(data_root, train_file_path, 416, 896)
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
                            {'params': model.pose_net.parameters(), 'lr': 3 * 1e-4}], lr=5 * 1e-4, weight_decay=5e-4)

    # only_pose = torch.optim.Adam(model.pose_net.parameters(), lr=5 * 1e-4, weight_decay=5e-4)
    # only_depth = torch.optim.Adam(model.depth_net.parameters(), lr=5 * 1e-3, weight_decay=5e-4)
    # opt = torch.optim.Adam([{'params': model.depth_net.parameters()}], lr=1e-3, weight_decay=5e-4)
    # sched = torch.optim.lr_scheduler.MultiStepLR(
    #     opt, milestones=[80, 160, 230, 180], gamma=0.5)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[66, 88, 166], gamma=0.5)

    g_loss = ReprojectLoss()
    s_loss = EdgeSmoothLoss()

    auto_mask = AutoMask()

    for idx, inputs in enumerate(train_loader):
        if idx < 12:
            continue
        for i in range(120):
            # 用原图推理获取相关的数据
            cur_image_0 = inputs['prime0_0']
            pre_image_0 = inputs['prime-1_0']
            next_image_0 = inputs['prime1_0']
            K0 = inputs['K_0']
            inv_K0 = inputs['inv_K0']
            # 参数变换
            depth_maps, refers_pose, next_pose = model(pre_image_0, cur_image_0, next_image_0)
            refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
            next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
            print(next_pose[..., 3:], refers_pose[..., 3:])
            total_loss = 0

            for scale in range(2, multi_scales):
                # monodepth2的训练方式
                # TODO: 输入图片一开始直接给个归一化来大力试试，不然loss爆炸
                cur_image = inputs['prime0_{}'.format(scale)]
                pre_image = inputs['prime-1_{}'.format(scale)]
                next_image = inputs['prime1_{}'.format(scale)]
                K = inputs['K_{}'.format(scale)]
                inv_K = inputs['inv_K{}'.format(scale)]
                # 暂时使用一个scale
                # 计算出投影的矩阵
                # TODO： 因为automask会去找多个图片的重投影最小的区域，
                #  1. 所以如果图中有比较多的黑色区域他们会倾向于填充该区域.
                #  2. 如果图中弱纹理的区域越大的话，mask的区域也会越大，本来他是为了解决occusion的问题反倒会成为姿态估计的干扰
                #  3. 如果不适用这个的话，那么物体和物体间的边缘偏差都会比较高

                # TODO: 因为弱纹理的区域太多。所以模型偷懒直接把边缘预测得比周边高，这样子在移动的时候就可以省力了
                # 怎么去给一个平面的约束，使得边缘的位置的深度个周边是一致的
                mask = auto_mask.compute_real_project(cur_image, pre_image, next_image)
                # 这里做一个scale up
                depth_map = depth_maps[scale]
                # TODO： 训练不起来，是不是要提高一下帧间差距
                # TODO： 既然训不起来，我觉得很大一个程度是因为分辨率的问题，小分辨率容易收敛，大分辨率很崩。跟我微调的分辨率应该是一个样子吧
                # TODO： 因为降采样以后占比的loss比较低，我直接给他们upscale试试
                # TODO: 1. 既然最上层的深度图有点难受，那么看看下层的深度图
                # 1. 如果固定好姿态输出的话应该是可以的,姿态估计的网络是比较难训练的
                _, depth = disp_to_depth(depth_map, min_depth, max_depth)
                # depth = depth_map
                # @ljq: 强制固定画面，看能不能学到姿态估计网络的样子
                # if i <= 40:
                #     depth = torch.ones(depth_map.size()) * 0.7
                refers_grid = get_sample_grid(depth, K, inv_K, refers_trans)
                next_grid = get_sample_grid(depth, K, inv_K, next_trans)

                pre2cur = view_syn(pre_image, refers_grid)
                next2cur = view_syn(next_image, next_grid)
                # visualTensor(pre2cur)
                # BBQ，
                pre_losses = auto_mask(pre2cur, cur_image, mask)
                next_losses = auto_mask(next2cur, cur_image, mask)
                current_loss = pre_losses.mean() + next_losses.mean() + s_weight * s_loss(depth_map, cur_image)
                total_loss += current_loss
                # 细节纹理来说，如果upsample的话很容易会遇到问题的，
                # TODO： 那么如果针对缩放度比较高的图片细节纹理会不会丢失很多
                print("第{}eoch，第{}层输出,loss为{}".format(i, scale, current_loss))
                # total_loss += (g_loss(pre2cur, cur_image) + g_loss(next2cur, cur_image)).mean() + s_weight * s_loss(
                #     depth_maps[-1], cur_image)
                # 因为Z轴没有动
                # loss = (g_loss(next2cur, cur_image)).mean() + s_weight * s_loss(depth_maps[-1], cur_image)

            # 加点多层深度图之间的一个约束，多个深度图之间要有一致性
            print(total_loss)
            # if i <= 40:
            #     only_pose.zero_grad()
            #     sum(total_loss[-2:]).backward()
            #     only_pose.step()
            # elif 40 < i <= 80:
            #     only_depth.zero_grad()
            #     sum(total_loss[-2:]).backward()
            #     only_depth.step()
            # else:
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            sched.step()

        VisualResult(model, inputs)
        # 转换参数
        break

# TODO:
#  1. 姿态估计网络参数发生变动，深度网络参数发生变动导致训练的不稳定性。
#  2. 跨物体边缘差异比较大，这应该也不是什么问题。物体块里面的差异比较小，导致一些low，texture会有点问题
#  3. 去掉auto_mask是不可行的，因为这样子会导致一些self.egomotion的物体不能被去除

# TODO:
# 1. 通过多尺度得一个训练，来解决了姿态估计不稳定不收敛得原因。
