# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 14:12
# @Author  : ljq
# @desc    : 
# @File    : TripleImages.py
import os
import time
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
# 导入损失函数
from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn, MonoPloter, MonoViewer
from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask
from Transfer.MonoDepth.MonoUtils.CameraTrans import transformation_from_parameters, get_sample_grid


class TripleTrainer:
    weight_decay = 5e-4
    epochs = 100
    # 使用torch amp单卡可以到达bs 10
    batch_size = 10
    num_workers = 4
    multi_scales = 4
    s_weight = 1e-3
    # min_depth 设置的太大的话也不行，STN不愿意动
    max_depth = 11
    # 收敛以后T值的大小跟min_depth是有关系的
    min_depth = 2

    def __init__(self, train_dataset, model, model_path='', is_parallel=False, **kwargs):
        # 这边固定好数值
        for k_name, value in kwargs.items():
            setattr(self, k_name, value)
        # 训练
        self.device = torch.device("cuda")
        self.model = model
        if len(model_path) > 0:
            self.resume_from(model_path)

        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=8)
        self.sample_loader = DataLoader(train_dataset, 1)

        self.opt = torch.optim.Adam([{'params': self.model.depth_net.parameters()},
                                     {'params': self.model.pose_net.parameters(), 'lr': 1e-4}], lr=5 * 1e-4,
                                    weight_decay=5e-4)
        # 分批更新梯度
        # self.pose_opt = torch.optim.Adam(self.model.pose_net.parameters(), lr=1e-4, weight_decay=5e-4)
        # self.depth_net_opt = torch.optim.Adam(self.model.depth_net.parameters(), lr=5e-4, weight_decay=5e-4)

        # 深度估计的质量很大程度取决于pose net的估计值
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[40, 80, 166], gamma=0.5)

        self.g_loss = ReprojectLoss()
        self.s_loss = EdgeSmoothLoss()
        # 一般用这个auto_mask来替代
        self.auto_mask = AutoMask()
        self.plotter = MonoPloter([self.min_depth, self.max_depth])
        self.is_parallel = is_parallel
        # 开个多卡
        if self.is_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        self.save_path = '/root/project/AwsomeDL/data/baseline'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.viewer = MonoViewer(self.model, self.min_depth, self.max_depth, False, True)
        self.scaler = GradScaler()

    def train(self):
        for e in range(self.epochs):
            total_loss = self.train_step()
            print('epoch is {}, train_loss is {}'.format(e, sum(total_loss) / (len(total_loss) + 1e-7)))

            # 测试的适合暂时不用
            # self.sched.step()
            if e % 30 == 0 or e % 100 == 0:
                self.save_checkpoint(e)
                # for idx, inputs in enumerate(self.sample_loader):
                #     for key, ipt in inputs.items():
                #         inputs[key] = ipt.to(self.device)
                #     self.model.eval()
                #     self.visual_result(inputs)
                #     break
            self.sched.step()

    def compute_loss(self, depth_maps, inputs, refers_trans, next_trans, cur_epoch=0, start_scale=0):
        total_loss = []
        # TODOL
        cur_images = [inputs['prime0_{}'.format(i)] for i in range(len(depth_maps))]
        pre_images = [inputs['prime-1_{}'.format(i)] for i in range(len(depth_maps))]
        next_images = [inputs['prime1_{}'.format(i)] for i in range(len(depth_maps))]

        # 这边计算auto mask

        for scale in range(start_scale, len(depth_maps)):
            # 取不同层次的图片
            trans_size = scale
            # if scale <= 1:
            #     # trans_size = scale + 1
            #     trans_size = scale + 2
            # else:
            #     trans_size = 3
            # if scale > 2:
            #     trans_size = 2

            cur_image = cur_images[trans_size]
            pre_image = pre_images[trans_size]
            next_image = next_images[trans_size]
            K = inputs['K_{}'.format(trans_size)]
            inv_K = inputs['inv_K{}'.format(trans_size)]

            depth_map = depth_maps[scale]
            # 对特征图进行上采样
            if trans_size != scale:
                depth_map = F.interpolate(depth_map, scale_factor=2 ** (scale - trans_size), mode='bilinear')

            _, depth = disp_to_depth(depth_map, self.min_depth, self.max_depth)
            refers_grid, _ = get_sample_grid(depth, K, inv_K, refers_trans)
            next_grid, _ = get_sample_grid(depth, K, inv_K, next_trans)
            # TODO: 姿态估计跳到0之后就起不来了，但是如果只是底层两层的特征图的话还是能起得来的

            pre2cur = view_syn(pre_image, refers_grid)
            next2cur = view_syn(next_image, next_grid)
            mask = self.auto_mask.compute_real_project(cur_image, pre_image, next_image)

            pre_losses = self.auto_mask(pre2cur, cur_image, mask)
            next_losses = self.auto_mask(next2cur, cur_image, mask)
            # 如果不加auto mask的话也根本训不起来的。

            # 底层监督高层
            # TODO： 因为一张图所以对全部做mean，之后肯定是在batch size的维度保持一直的。dim 应该是1
            current_loss = pre_losses + next_losses + self.s_weight * self.s_loss(
                depth_map, cur_image)
            # total_loss += current_loss
            total_loss.append(current_loss)
            # print("第{}epoch，第{}层输出,loss为{}".format(cur_epoch, scale, current_loss))
        # weights = [0.1, 0.2, 0.3, 0.4]
        # TODO: 动态的权重
        weights = [0.2, 0.2, 0.3, 0.3]

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

    def train_step(self):
        self.model.train()
        train_loss = []
        for idx, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            cur_image_0 = inputs['prime0_0']
            pre_image_0 = inputs['prime-1_0']
            next_image_0 = inputs['prime1_0']
            # 参数变换
            with autocast():
                self.opt.zero_grad()
                depth_maps, refers_pose, next_pose = self.model(pre_image_0, cur_image_0, next_image_0)

                geometry_loss = self.compute_depth_geometry_consistency(depth_maps)

                refers_trans = transformation_from_parameters(refers_pose[..., :3], refers_pose[..., 3:], True)
                next_trans = transformation_from_parameters(next_pose[..., :3], next_pose[..., 3:])
                depth_loss = self.compute_loss(depth_maps, inputs, refers_trans, next_trans)

                # 几何一致性+ 变换损失，强行约束最上层的深度图收敛
                # TODO: 深度图确实比较一致了，但是解决不了unstable的问题
                total_loss = (0.97 * depth_loss + 0.03 * geometry_loss).mean()
                train_loss.append(total_loss.detach().cpu().numpy())
                # 优化姿态的optimizer，计算姿态的loss，一般是底层比较大，高层比较小，大概是一个1/2的衰减
                # self.pose_opt.zero_grad()
                # self.depth_net_opt.zero_grad()
                # total_loss.backward()
                # self.opt.step()
                # torch amp
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            # self.pose_opt.step()
            # self.depth_net_opt.step()
        return train_loss

    def save_checkpoint(self, e):
        self.model.eval()
        save_name = "{}_model.pth".format(e)
        if self.is_parallel:
            model = self.model.module
        else:
            model = self.model
        torch.save(model.state_dict(), os.path.join(self.save_path, save_name))
        self.model.train()

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

            if idx < 70:
                continue
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            self.visual_result(inputs)
            break

    def recorder(self):
        self.model.eval()
        self.viewer.update_model(self.model)
        self.viewer.create_video_saver('/root/project/AwsomeDL/data/baseline')
        for idx, inputs in enumerate(self.sample_loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)
            self.viewer.save_video(inputs)
        self.viewer.stop_recorder()
