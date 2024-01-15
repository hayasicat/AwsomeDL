# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 14:12
# @Author  : ljq
# @desc    : 
# @File    : TripleImages.py
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 导入损失函数
from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn, MonoViewer
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
                                     {'params': model.pose_net.parameters(), 'lr':  1e-4}], lr=5 * 1e-4,
                                    weight_decay=5e-4)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[66, 88, 166], gamma=0.5)

        self.g_loss = ReprojectLoss()
        self.s_loss = EdgeSmoothLoss()
        # 一般用这个auto_mask来替代
        self.auto_mask = AutoMask()
        self.model = model
        self.viewer = MonoViewer([self.min_depth, self.max_depth])

    def train(self):
        for i in range(self.epochs):
            self.train_step()
            # 测试的适合暂时不用
            # self.sched.step()

    def compute_loss(self, depth_maps, inputs, refers_trans, next_trans, cur_epoch=0):
        total_loss = 0
        cur_images = [inputs['prime0_{}'.format(i)] for i in range(len(depth_maps))]
        pre_images = [inputs['prime-1_{}'.format(i)] for i in range(len(depth_maps))]
        next_images = [inputs['prime1_{}'.format(i)] for i in range(len(depth_maps))]

        for scale in range(0, len(depth_maps)):
            # 取不同层次的图片
            cur_image = cur_images[scale]
            pre_image = pre_images[scale]
            next_image = next_images[scale]
            K = inputs['K_{}'.format(scale)]
            inv_K = inputs['inv_K{}'.format(scale)]

            depth_map = depth_maps[scale]
            _, depth = disp_to_depth(depth_map, self.min_depth, self.max_depth)
            refers_grid = get_sample_grid(depth, K, inv_K, refers_trans)
            next_grid = get_sample_grid(depth, K, inv_K, next_trans)
            # TODO: 姿态估计跳到0之后就起不来了，但是如果只是底层两层的特征图的话还是能起得来的

            pre2cur = view_syn(pre_image, refers_grid)
            next2cur = view_syn(next_image, next_grid)
            mask = self.auto_mask.compute_real_project(cur_image, pre_image, next_image)
            pre_losses = self.auto_mask(pre2cur, cur_image, mask)
            next_losses = self.auto_mask(next2cur, cur_image, mask)

            # 底层监督高层
            # TODO： 因为一张图所以对全部做mean，之后肯定是在batch size的维度保持一直的。dim 应该是1
            current_loss = pre_losses.mean() + next_losses.mean() + self.s_weight * self.s_loss(depth_map, cur_image)
            res = 3
            # 套壳一层缩放看看
            # if scale < 3:
            #     cur_image = cur_images[res]
            #     pre_image = pre_images[res]
            #     next_image = next_images[res]
            #     K = inputs['K_{}'.format(res)]
            #     inv_K = inputs['inv_K{}'.format(res)]
            #
            #     down_factor = (0.5) ** (res - scale)
            #     down_sample_depth = F.interpolate(depth_map, scale_factor=down_factor, mode='nearest')
            #     _, depth = disp_to_depth(down_sample_depth, self.min_depth, self.max_depth)
            #     refers_grid = get_sample_grid(depth, K, inv_K, refers_trans)
            #     next_grid = get_sample_grid(depth, K, inv_K, next_trans)
            #     # TODO: 姿态估计跳到0之后就起不来了，但是如果只是底层两层的特征图的话还是能起得来的
            #     # TODO: 观察到的现象是姿态估计没有快速收敛和稳定以后很容易造成波动，然后带着深度预测一起波动。本来已经接近于0.3了谁聊到搏动了
            #     pre2cur = view_syn(pre_image, refers_grid)
            #     next2cur = view_syn(next_image, next_grid)
            #     mask = self.auto_mask.compute_real_project(cur_image, pre_image, next_image)
            #     pre_losses = self.auto_mask(pre2cur, cur_image, mask)
            #     next_losses = self.auto_mask(next2cur, cur_image, mask)
            #
            #     down_sample_loss = pre_losses.mean() + next_losses.mean() + self.s_weight * self.s_loss(
            #         down_sample_depth, cur_image)
            #     current_loss = 0.7 * current_loss + 0.3 * down_sample_loss
            total_loss += current_loss
            print("第{}epoch，第{}层输出,loss为{}".format(cur_epoch, scale, current_loss))
        return total_loss

    def compute_depth_geometry_consistency(self, depth_maps):
        """
        如果加上多个层级的loss会导致输出一直被抑制为0，导致更难收敛，本来是希望有个正向的作用，反倒是有往坏的发展
        :param depth_maps:
        :return:
        """
        depth_consistency_loss = 0
        for i in range(len(depth_maps) - 1):
            cur_depth = depth_maps[i]
            next_depth = depth_maps[i + 1]
            down_sample_depth = F.interpolate(cur_depth, scale_factor=0.5, mode='bilinear')
            # 限制一下
            # _, down_sample_depth = disp_to_depth(down_sample_depth, self.min_depth, self.max_depth)
            # _, next_depth = disp_to_depth(next_depth, self.min_depth, self.max_depth)

            cur_loss = (next_depth - down_sample_depth) ** 2
            depth_consistency_loss += cur_loss.mean()
            print("第{}层输出,loss为{}".format(i, cur_loss.mean()))
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
                total_loss = 0.97 * total_loss + 0.03 * geometry_loss
                print(total_loss)
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                # self.sched.step()
            self.visual_result(inputs)
            break

    def save_checkpoint(self):
        pass

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

        refers_grid = get_sample_grid(depth, K, inv_K, refers_trans)
        next_grid = get_sample_grid(depth, K, inv_K, next_trans)

        pre2cur = view_syn(pre_image, refers_grid)
        next2cur = view_syn(next_image, next_grid)

        # viewDepth(depth)
        self.viewer.show_image_tensor(next2cur * 255)
        self.viewer.show_image_tensor(cur_image * 255)
        self.viewer.show_image_tensor(next_image * 255)

        next_loss = self.auto_mask(next2cur, cur_image, mask)
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
