# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 14:12
# @Author  : ljq
# @desc    : 
# @File    : TripleImages.py
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 导入损失函数
from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn, MonoPloter
from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask
from Transfer.MonoDepth.MonoUtils.CameraTrans import transformation_from_parameters, get_sample_grid


class TripleTrainer():
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

        self.g_loss = ReprojectLoss()
        self.s_loss = EdgeSmoothLoss()
        # 一般用这个auto_mask来替代
        self.auto_mask = AutoMask()
        self.model = model
        self.viewer = MonoPloter([self.min_depth, self.max_depth])
        self.is_parallel = False
        self.save_path = '/root/project/AwsomeDL/data/monodepth'

    def train(self):
        for i in range(self.epochs):
            self.train_step()
            # 测试的适合暂时不用
            # self.sched.step()

    def compute_loss(self, depth_maps, inputs, refers_trans, next_trans, cur_epoch=0, start_scale=0):
        total_loss = []
        # TODOL
        weight_step_multi = 1.5
        cur_images = [inputs['prime0_{}'.format(i)] for i in range(len(depth_maps))]
        pre_images = [inputs['prime-1_{}'.format(i)] for i in range(len(depth_maps))]
        next_images = [inputs['prime1_{}'.format(i)] for i in range(len(depth_maps))]

        # TODO: 说明即使在底层对姿态估计训练到稳定了，加入上层还是变化幅度很大，实际上还是大尺度的特征图还是有问题的
        # start_idx = 0
        # if cur_epoch < 40:
        #     start_idx = 2

        for scale in range(start_scale, len(depth_maps)):
            # 取不同层次的图片
            cur_image = cur_images[scale]
            pre_image = pre_images[scale]
            next_image = next_images[scale]
            K = inputs['K_{}'.format(scale)]
            inv_K = inputs['inv_K{}'.format(scale)]

            depth_map = depth_maps[scale]
            _, depth = disp_to_depth(depth_map, self.min_depth, self.max_depth)
            refers_grid, _ = get_sample_grid(depth, K, inv_K, refers_trans)
            next_grid, _ = get_sample_grid(depth, K, inv_K, next_trans)
            # TODO: 姿态估计跳到0之后就起不来了，但是如果只是底层两层的特征图的话还是能起得来的

            pre2cur = view_syn(pre_image, refers_grid)
            next2cur = view_syn(next_image, next_grid)
            mask = self.auto_mask.compute_real_project(cur_image, pre_image, next_image)

            pre_losses = self.auto_mask(pre2cur, cur_image, mask)
            next_losses = self.auto_mask(next2cur, cur_image, mask)
            # 如果不加auto mask的话也根本训不起来的。很蛋疼
            # pre_losses = self.g_loss(pre2cur, cur_image).mean([1,2,3])
            # next_losses = self.g_loss(next2cur, cur_image).mean([1,2,3])
            # print(pre_losses.mean([1,2,3]))
            # 底层监督高层
            # TODO： 因为一张图所以对全部做mean，之后肯定是在batch size的维度保持一直的。dim 应该是1
            current_loss = pre_losses + next_losses + self.s_weight * self.s_loss(
                depth_map, cur_image)
            # total_loss += current_loss
            total_loss.append(current_loss)
            print("第{}epoch，第{}层输出,loss为{}".format(cur_epoch, scale, current_loss))
        result_loss = sum([l * idx * weight_step_multi for idx, l in enumerate(total_loss)])
        return result_loss

    def compute_depth_geometry_consistency(self, depth_maps):
        """
        :param depth_maps:
        :return:
        """
        depth_consistency_loss = 0
        # TODO: 我怀疑最上层的原因在于提前收束以后开始反复横跳导致的

        for i in range(len(depth_maps) - 1):
            cur_depth = depth_maps[i]
            next_depth = depth_maps[i + 1]
            down_sample_depth = F.interpolate(cur_depth, scale_factor=0.5, mode='bilinear')
            # 限制一下
            # _, down_sample_depth = disp_to_depth(down_sample_depth, self.min_depth, self.max_depth)
            # _, next_depth = disp_to_depth(next_depth, self.min_depth, self.max_depth)

            cur_loss = torch.abs(next_depth - down_sample_depth) / (next_depth + down_sample_depth)
            depth_consistency_loss += cur_loss.squeeze(1).mean(2).mean(1)
            print("第{}层输出,loss为{}".format(i, cur_loss.squeeze(1).mean([1, 2])))
        print("总的深度差异为{}".format(depth_consistency_loss))
        return depth_consistency_loss

    def train_step(self):
        for idx, inputs in enumerate(self.train_loader):
            if idx < 12:
                continue
            for i in range(100):
                cur_image_0 = inputs['prime0_0']
                pre_image_0 = inputs['prime-1_0']
                next_image_0 = inputs['prime1_0']

                # 参数变换
                depth_maps, refers_pose, next_pose = self.model(pre_image_0, cur_image_0, next_image_0)

                print(next_pose[..., 3:], refers_pose[..., 3:])
                geometry_loss = self.compute_depth_geometry_consistency(depth_maps)

                refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
                next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
                total_loss = self.compute_loss(depth_maps, inputs, refers_trans, next_trans, i)

                print(total_loss)
                # 几何一致性+ 变换损失，强行约束最上层的深度图收敛
                # TODO: 深度图确实比较一致了，但是解决不了unstable的问题
                total_loss = 0.95 * total_loss + 0.05 * geometry_loss
                print(total_loss)
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                # self.sched.step()
            self.visual_result(inputs)
            self.save_checkpoint(10)
            break

    def save_checkpoint(self, e):
        self.model.eval()
        save_name = "{}_model.pth".format(e)
        if self.is_parallel:
            model = self.model.module
        else:
            model = self.model
        torch.save(model.state_dict(), os.path.join(self.save_path, save_name))

    def resume_from(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        print("resum from {}".format(model_path))

    def visual_result(self, input_info):
        cur_image = input_info['prime0_0']
        pre_image = input_info['prime-1_0']
        next_image = input_info['prime1_0']
        K = input_info['K_0']
        inv_K = input_info['inv_K0']
        depth_maps, refers_pose, next_pose = self.model(pre_image, cur_image, next_image)
        refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
        next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
        mask = self.auto_mask.compute_real_project(cur_image, pre_image, next_image)
        _, depth = disp_to_depth(depth_maps[0], self.min_depth, self.max_depth)

        refers_grid, _ = get_sample_grid(depth, K, inv_K, refers_trans)
        next_grid, _ = get_sample_grid(depth, K, inv_K, next_trans)

        pre2cur = view_syn(pre_image, refers_grid)
        next2cur = view_syn(next_image, next_grid)

        # viewDepth(depth)
        self.viewer.show_image_tensor(next2cur * 255)
        self.viewer.show_image_tensor(cur_image * 255)
        self.viewer.show_image_tensor(next_image * 255)

        next_loss, idx = self.auto_mask.analyze(next2cur, cur_image, mask)
        mean_loss = next_loss.mean(2).mean(1)
        loss_map_from = F.one_hot(idx, num_classes=3) * 255
        loss_map_from = loss_map_from.squeeze(0).permute(2, 0, 1)

        mask = torch.min(mask, dim=1)[0] < 0.03

        self.viewer.show_reproject_loss(mask)

        self.viewer.show_reproject_loss(loss_map_from)
        self.viewer.show_reproject_loss(next_loss)

        next_loss = self.g_loss(next2cur, cur_image)

        self.viewer.show_reproject_loss(next_loss)

        # next_loss = g_loss(next_image_0, cur_image_0)
        # visualReproject(next_loss)

        # 干脆把所有的深度图给可视化把，如果底层的深度图不会变成桶，那么可以用底层的深度图来收敛商城的深度图
        for dps in depth_maps:
            _, dp = disp_to_depth(dps, self.min_depth, self.max_depth)
            print(dp.mean())
            self.viewer.show_depth(dp)
            time.sleep(0.1)
        print(next_pose, next_trans)
        print(refers_pose.size(), next_pose.size())

    def analys(self):
        for idx, inputs in enumerate(self.train_loader):
            if idx < 12:
                continue
            self.visual_result(inputs)
            break
