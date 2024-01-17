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
from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn, MonoPloter, MonoViewer
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
    min_depth = 2

    def __init__(self, train_dataset, model, model_path='', **kwargs):
        # 这边固定好数值
        for k_name, value in kwargs.items():
            setattr(self, k_name, value)
        # 训练
        self.device = torch.device("cuda")
        self.model = model
        if len(model_path) > 0:
            self.resume_from(model_path)
        self.model = self.model.to(self.device)

        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)

        self.sample_loader = DataLoader(train_dataset, self.batch_size)

        self.opt = torch.optim.Adam([{'params': self.model.depth_net.parameters()},
                                     {'params': self.model.pose_net.parameters(), 'lr': 1e-4}], lr=5 * 1e-4,
                                    weight_decay=5e-4)
        # 分批更新梯度
        self.pose_opt = torch.optim.Adam(self.model.pose_net.parameters(), lr=1e-4, weight_decay=5e-4)
        self.depth_net_opt = torch.optim.Adam(self.model.depth_net.parameters(), lr=5e-4, weight_decay=5e-4)

        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[66, 88, 166], gamma=0.5)

        self.g_loss = ReprojectLoss()
        self.s_loss = EdgeSmoothLoss()
        # 一般用这个auto_mask来替代
        self.auto_mask = AutoMask()
        self.plotter = MonoPloter([self.min_depth, self.max_depth])
        self.is_parallel = False
        self.save_path = '/root/project/AwsomeDL/data/monodepth'
        self.viewer = MonoViewer(self.model, self.min_depth, self.max_depth, False, True)

    def train(self):
        for i in range(self.epochs):
            self.train_step()
            # 测试的适合暂时不用
            # self.sched.step()

    def compute_loss(self, depth_maps, inputs, refers_trans, next_trans, cur_epoch=0, start_scale=2):
        total_loss = []
        # TODOL
        cur_images = [inputs['prime0_{}'.format(i)] for i in range(len(depth_maps))]
        pre_images = [inputs['prime-1_{}'.format(i)] for i in range(len(depth_maps))]
        next_images = [inputs['prime1_{}'.format(i)] for i in range(len(depth_maps))]
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

            # 底层监督高层
            # TODO： 因为一张图所以对全部做mean，之后肯定是在batch size的维度保持一直的。dim 应该是1
            current_loss = pre_losses + next_losses + self.s_weight * self.s_loss(
                depth_map, cur_image)
            # total_loss += current_loss
            total_loss.append(current_loss)
            print("第{}epoch，第{}层输出,loss为{}".format(cur_epoch, scale, current_loss))
        return sum(total_loss)

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
            # _, down_sample_depth = disp_to_depth(down_sample_depth, self.min_depth, self.max_depth)
            # _, next_depth = disp_to_depth(next_depth, self.min_depth, self.max_depth)
            cur_loss = torch.abs(down_sample_depth - next_depth) / (down_sample_depth + next_depth)
            depth_consistency_loss.append(cur_loss.squeeze(1).mean([1, 2]))
            print("第{}层输出,loss为{}".format(i, cur_loss.squeeze(1).mean([1, 2])))
        return sum(depth_consistency_loss)

    def train_step(self):
        for i in range(100):
            self.model.train()
            for idx, inputs in enumerate(self.sample_loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                if idx < 12:
                    continue

                cur_image_0 = inputs['prime0_0']
                pre_image_0 = inputs['prime-1_0']
                next_image_0 = inputs['prime1_0']

                # 参数变换
                depth_maps, refers_pose, next_pose = self.model(pre_image_0, cur_image_0, next_image_0)

                print(next_pose[..., 3:], refers_pose[..., 3:])
                geometry_loss = self.compute_depth_geometry_consistency(depth_maps)

                refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
                next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
                depth_loss = self.compute_loss(depth_maps, inputs, refers_trans, next_trans, i)

                print(depth_loss)
                # 几何一致性+ 变换损失，强行约束最上层的深度图收敛
                # TODO: 深度图确实比较一致了，但是解决不了unstable的问题
                total_loss = 0.85 * depth_loss + 0.03 * geometry_loss
                # 优化姿态的optimizer，计算姿态的loss，一般是底层比较大，高层比较小，大概是一个1/2的衰减

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                break
            if i % 30 == 0:
                self.save_checkpoint('temp_' + str(i))
                # self.sched.step()
        self.save_checkpoint(10)
        for idx, inputs in enumerate(self.sample_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            if idx < 12:
                continue
            self.model.eval()
            self.visual_result(inputs)

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
        self.model.to(self.device)
        print("resum from {}".format(model_path))

    def visual_result(self, input_info):
        self.model.eval()
        self.viewer.update_model(self.model)
        self.viewer.visual_syn_image(input_info, False, True, start_scale=0)

    def analys(self):
        for idx, inputs in enumerate(self.sample_loader):

            if idx < 12:
                continue
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            self.visual_result(inputs)
            break
