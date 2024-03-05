# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 14:12
# @Author  : ljq
# @desc    : 
# @File    : PairImages.py
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn, MonoPloter, MonoViewer
from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask
from Transfer.MonoDepth.MonoUtils.CameraTrans import transformation_from_parameters, get_sample_grid
from Tools.Logger.my_logger import init_logger


class PairTrainer():
    weight_decay = 5e-4
    epochs = 100
    batch_size = 4
    num_workers = 4
    multi_scales = 4
    s_weight = 1e-2
    # min_depth 设置的太大的话也不行，STN不愿意动
    max_depth = 11
    # 收敛以后T值的大小跟min_depth是有关系的
    min_depth = 2

    def __init__(self, train_dataset, model, model_path='', **kwargs):
        # 这边固定好数值
        for k_name, value in kwargs.items():
            setattr(self, k_name, value)
        # 训练
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=8)
        self.sample_loader = DataLoader(train_dataset, 1)
        self.opt = torch.optim.Adam([{'params': model.depth_net.parameters()},
                                     {'params': model.pose_net.parameters(), 'lr': 1e-4}], lr=8 * 1e-4,
                                    weight_decay=5e-4)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[50, 80, 166], gamma=0.5)

        # 位姿网络和
        self.pose_opt = torch.optim.Adam(model.pose_net.parameters(), lr=1e-4, weight_decay=5e-4)
        self.depth_opt = torch.optim.Adam(model.depth_net.parameters(), lr=5 * 1e-4, weight_decay=5e-4)
        self.pose_sched = torch.optim.lr_scheduler.MultiStepLR(self.pose_opt, milestones=[30, 50, 75], gamma=0.5)
        self.depth_sched = torch.optim.lr_scheduler.MultiStepLR(self.depth_opt, milestones=[40, 60, 80], gamma=0.5)

        # 这两个loss为主
        self.reproject_l = ReprojectLoss()
        self.edge_smooth_l = EdgeSmoothLoss()
        # 一般用这个auto_mask来替代
        self.auto_mask = AutoMask()
        self.device = torch.device("cuda")

        self.model = model
        self.model = self.model.to(self.device)
        # self.viewer = MonoPloter([self.min_depth, self.max_depth])
        # 限制
        self.viewer = MonoViewer(self.model, self.min_depth, self.max_depth, using_auto_mask=True, using_sc_depth=True)
        self.is_parallel = False
        self.save_path = '/root/project/AwsomeDL/data/sc_depth'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.scaler = GradScaler()

        if len(model_path) > 0:
            self.resume_from(model_path)

        init_logger('/root/project/AwsomeDL/data/logs/pair_image_train.log')
        self.logger = logging.getLogger('train')

    def train(self):
        for e in range(self.epochs):
            total_loss = self.train_step(e)
            self.logger.info('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / (len(total_loss) + 1e-7)))
            if e % 30 == 0 or e % 100 == 0:
                self.save_checkpoint(e)
            # 测试的适合暂时不用
            # self.sched.step()
            self.pose_sched.step()
            self.depth_sched.step()

    def compute_loss(self, depth_maps, poses, inputs, weights=[0.3, 0.3, 0.2, 0.2]):
        # 重建图片，然后计算loss
        total_loss = []
        refers_pose, next_pose, cur_pose = poses
        cur_depth_map, pre_depth_map, nex_depth_map = depth_maps

        cur_images = [inputs['prime0_{}'.format(i)] for i in range(len(cur_depth_map))]
        pre_images = [inputs['prime-1_{}'.format(i)] for i in range(len(cur_depth_map))]
        next_images = [inputs['prime1_{}'.format(i)] for i in range(len(cur_depth_map))]

        cur2pre = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
        # pre_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:])

        cur2next = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])

        pre2cur = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:])
        next2cur = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:], True)
        # 直接通过前一张图片重建后一张图片
        pre2next = torch.matmul(pre2cur, cur2next)

        # 算一个直接pre到

        # 获取auto mask
        for scale in range(0, len(cur_depth_map)):
            trans_size = scale
            # if trans_size > 1:
            #     trans_size = 1

            K = inputs['K_{}'.format(trans_size)]
            inv_K = inputs['inv_K{}'.format(trans_size)]
            cur_img = cur_images[trans_size]
            pre_img = pre_images[trans_size]
            next_img = next_images[trans_size]

            # 计算mask
            mask = self.auto_mask.compute_real_project(cur_img, pre_img, next_img)

            # 相邻的图片为基准,投影图片
            _, cur_depth = disp_to_depth(cur_depth_map[scale], self.min_depth, self.max_depth)
            _, pre_depth = disp_to_depth(pre_depth_map[scale], self.min_depth, self.max_depth)
            _, next_depth = disp_to_depth(nex_depth_map[scale], self.min_depth, self.max_depth)
            if trans_size != scale:
                # 进行缩放
                cur_depth = F.interpolate(cur_depth, scale_factor=2 ** (scale - trans_size), mode='bilinear')
                pre_depth = F.interpolate(pre_depth, scale_factor=2 ** (scale - trans_size), mode='bilinear')
                next_depth = F.interpolate(next_depth, scale_factor=2 ** (scale - trans_size), mode='bilinear')

            photo_loss_p2c, diff_depth_p2c = compute_auto_mask_loss(cur_img, pre_img, cur_depth, pre_depth,
                                                                    cur2pre, K, inv_K, self.auto_mask, mask)

            photo_loss_n2c, diff_depth_n2c = compute_auto_mask_loss(cur_img, next_img, cur_depth, next_depth,
                                                                    cur2next, K, inv_K, self.auto_mask, mask)
            # 加一下重投影误差
            # photo_loss_c2p, diff_depth_c2p = compute_auto_mask_loss(pre_img, cur_img, pre_depth, cur_depth,
            #                                                         pre2cur, K, inv_K, self.auto_mask, mask)
            # photo_loss_c2n, diff_depth_c2n = compute_auto_mask_loss(next_img, cur_img, next_depth, cur_depth,
            #                                                         next2cur, K, inv_K, self.auto_mask, mask)

            # photo_loss_p2n, diff_depth_p2n = compute_auto_mask_loss(pre_img, next_img, pre_depth, next_depth,
            #                                                         pre2next, K, inv_K, self.auto_mask, mask)
            # print(photo_loss_p2c, photo_loss_p2n, diff_depth_p2n, diff_depth_p2c)
            photo_loss = (photo_loss_p2c + photo_loss_n2c)
            diff_depth = (diff_depth_n2c + diff_depth_p2c)
            # print(photo_loss_n2c.size(), diff_depth_n2c.mean([1, 2, 3]).size(),diff_depth_n2c.mean([1, 2, 3]))
            # print(diff_depth_n2c.mean([1, 2, 3]), photo_loss_p2c)
            smooth_l = self.edge_smooth_l(cur_depth, cur_img)

            # 加上一致性损失看看结果
            # current_loss = photo_loss_p2c + photo_loss_n2c + self.s_weight * smooth_l + \
            #                0.01 * diff_depth_n2c + 0.01 * diff_depth_p2c
            current_loss = photo_loss + self.s_weight * smooth_l + 0.01 * diff_depth
            total_loss.append(current_loss)
        # 通过不同的权重来解决这个问题
        # print(total_loss)
        # return sum(total_loss)
        return sum([w * l for w, l in zip(weights, total_loss)])

    def compute_depth_geometry_consistency(self, depth_maps):
        """
        :param depth_maps:
        :return:
        """
        depth_consistency_loss = []
        # TODO: 我怀疑最上层的原因在于提前收束以后开始反复横跳导致的
        for i in range(len(depth_maps) - 1):
            cur_depth = depth_maps[i]
            next_depth = depth_maps[i + 1]
            down_sample_depth = F.interpolate(cur_depth, scale_factor=0.5, mode='bilinear')
            # 限制一下
            _, down_sample_depth = disp_to_depth(down_sample_depth, self.min_depth, self.max_depth)
            _, next_depth = disp_to_depth(next_depth, self.min_depth, self.max_depth)
            cur_loss = torch.abs(down_sample_depth - next_depth) / (down_sample_depth + next_depth)
            depth_consistency_loss.append(cur_loss.squeeze(1).mean([1, 2]))
            # print("第{}层输出,loss为{}".format(i, cur_loss.squeeze(1).mean([1, 2])))
        return sum(depth_consistency_loss)

    def train_step(self, e=0):
        self.model.train()

        # if e < 30:
        #     weights = [0.1, 0.1, 0.3, 0.5]
        # else:
        #     weights = [0.3, 0.3, 0.2, 0.2]
        # 位姿能训练稳定的话就迅速去衰减他
        train_loss = []
        for idx, inputs in enumerate(self.train_loader):
            # if idx < 13:
            #     continue
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            cur_image_0 = inputs['prime0_0aug']
            pre_image_0 = inputs['prime-1_0aug']
            next_image_0 = inputs['prime1_0aug']

            with autocast():
                self.opt.zero_grad()
                # 参数变换
                depth_maps, poses = self.model.multi_depth(pre_image_0, cur_image_0, next_image_0)
                # 给隔开看看
                cur_depth_maps, pre_depth_maps, nex_depth_maps = depth_maps

                # total_loss = self.compute_loss(cur_depth_map, refers_depth_maps, inputs, pose_trans, inv_trans)
                total_loss = self.compute_loss(depth_maps, poses, inputs)

                geometry_loss = self.compute_depth_geometry_consistency(cur_depth_maps)
                # 计算一下余弦距离

                # 几何一致性+ 变换损失，强行约束最上层的深度图收敛
                # TODO: 深度图确实比较一致了，但是解决不了unstable的问题
                total_loss = (0.97 * total_loss + 0.03 * geometry_loss)
                total_loss = total_loss.mean()
                train_loss.append(total_loss.detach().cpu().numpy())
                if e <= 15:
                    # 看看是不是不同一个方向就GG了啊
                    shift_loss = self.compute_pose_pseudo_loss(inputs, poses)
                    if len(shift_loss) > 0:
                        total_loss = 0.3 * sum(shift_loss) / len(shift_loss) + total_loss

                self.scaler.scale(total_loss).backward()
                # self.scaler.step(self.opt)
                self.scaler.step(self.pose_opt)
                self.scaler.step(self.depth_opt)
                self.scaler.update()
            # break
        return train_loss

    def compute_pose_pseudo_loss(self, inputs, poses):
        key_names = list(inputs.keys())
        refers_pose, next_pose, cur_pose = poses
        # 判断一下有没有伪标签，有的话就算一下，没有的话就拜拜
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

        shift_loss = []
        if 'pre_t' in key_names and 'next_t' in key_names:
            # 有的话就计算咯
            pose_net_pre = refers_pose[..., 3:]
            pose_net_next = next_pose[..., 3:]
            pre_pseudo = inputs['pre_t']
            next_pseudo = inputs['next_t']
            # 来判断一下那些需要计算mask一下
            pre_sim = similarity(pose_net_pre, pre_pseudo)
            next_sim = similarity(pose_net_next, next_pseudo)
            mask = pre_pseudo.abs().sum(dim=[1, 2]) > 0.1
            pre_sim = pre_sim[mask]
            next_sim = next_sim[mask]
            pre_sim = pre_sim[pre_sim < 0.85]
            next_sim = next_sim[next_sim < 0.85]
            # print(pre_sim.size(0))
            # print(pose_net_pre, pre_pseudo)
            if pre_sim.size(0) > 0:
                shift_loss.append(1 - pre_sim.mean())
            if next_sim.size(0) > 0:
                shift_loss.append(1 - next_sim.mean())

        # 强制看一下自己和自己的Pose究竟多少
        # neg_loss = cur_pose[..., 3:].abs().sum(dim=[1, 2]).mean()
        # print(neg_loss)
        # if neg_loss > 0.1:
        #     shift_loss.append(0.1 * neg_loss)
        return shift_loss

    def save_checkpoint(self, e):
        # 用来保存最优的
        self.model.eval()
        save_name = "{}_model.pth".format(e)
        if self.is_parallel:
            model = self.model.module
        else:
            model = self.model
        torch.save(model.state_dict(), os.path.join(self.save_path, save_name))
        self.model.train()

    def visual_result(self, inputs):
        self.viewer.update_model(self.model)
        self.viewer.visual_syn_image(inputs, is_show_pred=False, is_show_next=True, start_scale=0)

    def analys(self):
        for idx, inputs in enumerate(self.sample_loader):
            if idx < 50:
                continue
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            self.visual_result(inputs)
            break

    def eval(self):
        self.model.eval()
        for idx, inputs in enumerate(self.sample_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            cur_image_0 = inputs['prime0_0']
            pre_image_0 = inputs['prime-1_0']
            next_image_0 = inputs['prime1_0']
            # 参数变换
            depth_maps, poses = self.model.multi_depth(pre_image_0, cur_image_0, next_image_0)
            # 给隔开看看
            cur_depth_maps, pre_depth_maps, nex_depth_maps = depth_maps
            total_loss = self.compute_loss(depth_maps, poses, inputs)
            print(total_loss, poses[0])

    def resume_from(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print("resum from {}".format(model_path))


reproject_loss = ReprojectLoss()
edge_smooth_loss = EdgeSmoothLoss()


def inverse_warp(ref_img, target_depth, ref_depth, trans, K, inv_K):
    """
    :param ref_img:
    :param target_depth:
    :param ref_depth:
    :param trans:
    :param K:
    :param inv_K:
    :return:
    """
    refers_grid, compute_depth = get_sample_grid(target_depth, K, inv_K, trans)
    project_img = view_syn(ref_img, refers_grid)
    project_depth = view_syn(ref_depth, refers_grid)
    return project_img, project_depth, compute_depth, refers_grid


def compute_pair_loss(cur_img, refer_img, cur_depth, refer_depth, pose_trans, K, inv_K):
    syn_cur, syn_cur_depth, compute_depth, grid = inverse_warp(refer_img, cur_depth, refer_depth, pose_trans, K,
                                                               inv_K)
    # 计算深度差
    diff_depth = (compute_depth - syn_cur_depth).abs() / (compute_depth + syn_cur_depth)
    # 计算掩码
    weight_mask = (1 - diff_depth)
    # 计算loss
    photo_loss = reproject_loss(syn_cur, cur_img)
    photo_loss *= weight_mask
    return photo_loss, diff_depth


def compute_auto_mask_loss(cur_img, refer_img, cur_depth, refer_depth, pose_trans, K, inv_K, loss_backend, mask):
    syn_cur, syn_cur_depth, compute_depth, grid = inverse_warp(refer_img, cur_depth, refer_depth, pose_trans, K,
                                                               inv_K)
    diff_depth = (compute_depth - syn_cur_depth).abs() / (compute_depth + syn_cur_depth)
    diff_depth = diff_depth.mean([1])
    # photo_loss = loss_backend(syn_cur, cur_img, mask, using_mean=True)
    weight_mask = (1 - diff_depth)
    # 计算个grid的非mask的数量
    valid_mask = (grid[..., 0] < -1) | (grid[..., 0] > 1) | \
                 (grid[..., 1] < -1) | (grid[..., 1] > 1)
    valid_mask = 1 - valid_mask.float()
    # 用来控制相机静止但是到处飞得物体
    photo_loss = loss_backend(syn_cur, cur_img, mask, using_mean=False)
    photo_loss = valid_mask * photo_loss
    photo_loss = photo_loss.sum(dim=[1, 2]) / valid_mask.sum(dim=[1, 2])
    diff_depth = diff_depth.sum(dim=[1, 2]) / valid_mask.sum(dim=[1, 2])
    # photo_loss = (weight_mask * photo_loss).mean([1, 2])
    return photo_loss, diff_depth


def compute_origin_loss(cur_img, refer_img, cur_depth, refer_depth, pose_trans, K, inv_K):
    refers_grid, compute_depth = get_sample_grid(cur_depth, K, inv_K, pose_trans)
    project_img = view_syn(refer_img, refers_grid)
    photo_loss = reproject_loss(project_img, cur_img)
    return photo_loss
