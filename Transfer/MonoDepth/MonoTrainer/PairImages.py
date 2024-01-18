# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 14:12
# @Author  : ljq
# @desc    : 
# @File    : PairImages.py
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn, MonoPloter, MonoViewer
from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask
from Transfer.MonoDepth.MonoUtils.CameraTrans import transformation_from_parameters, get_sample_grid


class PairTrainer():
    weight_decay = 5e-4
    epochs = 1
    batch_size = 1
    num_workers = 4
    multi_scales = 4
    s_weight = 1e-3
    # min_depth 设置的太大的话也不行，STN不愿意动
    max_depth = 11
    # 收敛以后T值的大小跟min_depth是有关系的
    min_depth = 3

    def __init__(self, train_dataset, model, **kwargs):
        # 这边固定好数值
        for k_name, value in kwargs.items():
            setattr(self, k_name, value)
        # 训练
        self.train_loader = DataLoader(train_dataset, self.batch_size)
        self.opt = torch.optim.Adam([{'params': model.depth_net.parameters()},
                                     {'params': model.pose_net.parameters(), 'lr': 1e-4}], lr=5 * 1e-4,
                                    weight_decay=5e-4)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[66, 88, 166], gamma=0.5)
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
        self.viewer = MonoViewer(self.model, self.min_depth, self.max_depth, using_sc_depth=True)
        self.is_parallel = False
        self.save_path = '/root/project/AwsomeDL/data/sc_depth'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        for i in range(self.epochs):
            self.train_step()
            # 测试的适合暂时不用
            # self.sched.step()

    def compute_loss(self, cur_depth_map, refers_depth_maps, inputs, pose_trans, inv_trans):
        # 重建图片，然后计算loss
        total_loss = []
        cur_images = [inputs['prime0_{}'.format(i)] for i in range(len(cur_depth_map))]
        pre_images = [inputs['prime-1_{}'.format(i)] for i in range(len(cur_depth_map))]
        next_images = [inputs['prime1_{}'.format(i)] for i in range(len(cur_depth_map))]

        refers_images = [pre_images, next_images]
        for scale in range(3, len(cur_depth_map)):

            K = inputs['K_{}'.format(scale)]
            inv_K = inputs['inv_K{}'.format(scale)]
            cur_img = cur_images[scale]
            photo_losses = []
            diff_depth_losses = []
            smooth_losses = []

            for idx, pose_t, inv_pose_t in zip(range(len(pose_trans)), pose_trans, inv_trans):
                # 将深度图给缩放到准确的区域
                _, cur_depth = disp_to_depth(cur_depth_map[scale], self.min_depth, self.max_depth)
                _, refer_depth = disp_to_depth(refers_depth_maps[idx][scale], self.min_depth, self.max_depth)
                # 用来计算
                refer_img = refers_images[idx][scale]
                # 计算变换
                photo_loss_r2c, diff_depth_r2c = compute_pair_loss(cur_img, refer_img, cur_depth, refer_depth, pose_t,
                                                                   K, inv_K)

                # photo_loss_r2c, diff_depth_r2c = compute_pair_loss(cur_img, refer_img, cur_depth, refer_depth, pose_t,
                #                                                    K, inv_K)
                photo_losses.append(photo_loss_r2c)
                diff_depth_losses.append(diff_depth_r2c)

                # photo_loss_c2r, diff_depth_c2r = compute_pair_loss(refer_img, cur_img, refer_depth, cur_depth,
                #                                                    inv_pose_t,
                #                                                    K, inv_K)
                # photo_losses.append(photo_loss_c2r)
                # diff_depth_losses.append(diff_depth_c2r)

                smooth_losses.append(self.edge_smooth_l(cur_depth_map[scale], cur_img))
            # 保持当前的批梯度
            photo_l = torch.cat(photo_losses, dim=1).mean([1, 2, 3])
            diff_l = torch.cat(diff_depth_losses, dim=1).mean([1, 2, 3])
            smooth_l = torch.cat(smooth_losses, dim=1).mean([1])
            # print(photo_l, diff_l, smooth_l)
            total_loss.append(photo_l + self.s_weight * smooth_l + 0.1 * diff_l)

        return sum(total_loss)

    def train_step(self):
        for idx, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            if idx < 36:
                continue
            for i in range(100):
                cur_image_0 = inputs['prime0_0']
                pre_image_0 = inputs['prime-1_0']
                next_image_0 = inputs['prime1_0']

                # 参数变换
                refers_images = [pre_image_0]
                cur_depth_map, refers_depth_maps, pose_forward, pose_inv = self.model(cur_image_0, refers_images)

                pose_trans = [transformation_from_parameters(p[..., :3], p[..., 3:]) for p in pose_forward]
                inv_trans = [transformation_from_parameters(p[..., :3], p[..., 3:]) for p in pose_inv]

                print([p for p in pose_forward])
                total_loss = self.compute_loss(cur_depth_map, refers_depth_maps, inputs, pose_trans, inv_trans)
                # 几何一致性+ 变换损失，强行约束最上层的深度图收敛
                # TODO: 深度图确实比较一致了，但是解决不了unstable的问题
                total_loss = total_loss
                print(total_loss)
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                # self.sched.step()
            self.visual_result(inputs)
            self.save_checkpoint(20)
            break

    def save_checkpoint(self, e):
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
        self.viewer.visual_syn_image(inputs, is_show_next=False, start_scale=2)

    def analys(self):
        for idx, inputs in enumerate(self.train_loader):
            if idx < 36:
                continue
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            self.visual_result(inputs)
            break

    def resume_from(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
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
    return project_img, project_depth, compute_depth


def compute_pair_loss(cur_img, refer_img, cur_depth, refer_depth, pose_trans, K, inv_K):
    syn_cur, syn_cur_depth, compute_depth = inverse_warp(refer_img, cur_depth, refer_depth, pose_trans, K,
                                                         inv_K)
    # 计算深度差
    diff_depth = (compute_depth - syn_cur_depth).abs() / (compute_depth + syn_cur_depth)
    # 计算掩码
    weight_mask = (1 - diff_depth)
    print(weight_mask.sum())
    # 计算loss
    photo_loss = reproject_loss(syn_cur, cur_img)
    photo_loss *= weight_mask
    return photo_loss, diff_depth


def compute_origin_loss(cur_img, refer_img, cur_depth, refer_depth, pose_trans, K, inv_K):
    refers_grid, compute_depth = get_sample_grid(cur_depth, K, inv_K, pose_trans)
    project_img = view_syn(refer_img, refers_grid)
    photo_loss = reproject_loss(project_img, cur_img)
    return photo_loss
