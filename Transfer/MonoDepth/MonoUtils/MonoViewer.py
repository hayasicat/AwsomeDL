# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 15:54
# @Author  : ljq
# @desc    : 
# @File    : MonoViewer.py
import torch
import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from Transfer.MonoDepth.MonoUtils import disp_to_depth, view_syn
from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask
from Transfer.MonoDepth.MonoUtils.CameraTrans import transformation_from_parameters, get_sample_grid


class MonoPloter():
    def __init__(self, depth_range=[1, 10]):
        self.depth_range = depth_range
        pass

    def show_depth(self, depth, is_show_normal=True):
        if len(depth.size()) > 3:
            depth = depth[0, ...]
        depth = depth.detach().cpu().permute(1, 2, 0).numpy()
        if is_show_normal:
            vmax = np.percentile(depth, 95)
            normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            render_image = (mapper.to_rgba(depth.reshape((depth.shape[0], depth.shape[1])))[:, :, :3] * 255).astype(
                np.uint8)
        else:
            depth = depth.reshape((depth.shape[0], depth.shape[1]))
            # render_image = np.ones_like(depth)
            # 进行着色显示
            depth -= self.depth_range[0]
            render_image = depth / float(self.depth_range[1] - self.depth_range[0]) * 255
            render_image.astype(np.uint8)
        self.show(render_image)
        return render_image

    def show_reproject_loss(self, loss_tensor):
        # 重投影的loss可视化
        if len(loss_tensor.size()) > 3:
            loss_tensor = loss_tensor[0, ...]
        loss_tensor = loss_tensor.detach().cpu().permute(1, 2, 0).numpy()
        self.show(loss_tensor)
        return loss_tensor

    def show_image_tensor(self, image_tensor):
        # batchsize就一第一张
        if len(image_tensor.size()) > 3:
            image_tensor = image_tensor[0, ...]
        new_image = image_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        self.show(new_image)
        return new_image

    def show(self, image):
        plt.imshow(image)
        plt.show()


class MonoViewer():
    def __init__(self, model, min_depth=3, max_depth=10, using_sc_depth=False):
        self.model = model
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ploter = MonoPloter([min_depth, max_depth])
        self.multi_scale = 4
        self.using_sc_depth = using_sc_depth

    def visual_syn_image(self, input_info, is_show_pred=True, is_show_next=True, start_scale=0):
        #
        cur_images = [input_info['prime0_{}'.format(s)] for s in range(self.multi_scale)]
        pre_images = [input_info['prime-1_{}'.format(s)] for s in range(self.multi_scale)]
        next_images = [input_info['prime1_{}'.format(s)] for s in range(self.multi_scale)]
        Ks = [input_info['K_{}'.format(s)] for s in range(self.multi_scale)]
        inv_Ks = [input_info['inv_K{}'.format(s)] for s in range(self.multi_scale)]

        target_x = (cur_images[0] - 0.45) / 0.225
        source_x = []
        source_images = []

        if is_show_pred:
            source_x.append((pre_images[0] - 0.45) / 0.225)
            source_images.append(pre_images)
        if is_show_next:
            source_x.append((next_images[0] - 0.45) / 0.225)
            source_images.append(next_images)

        # 预测深度图和相机姿态
        depth_maps = self.model.depth_map(target_x)
        # 如果使用的是sc_depth方法
        refer_depths = []

        cam_poses = []
        cam_trans = []
        for source_ in source_x:
            pose = self.model.get_pose(target_x, source_)
            if self.using_sc_depth:
                refer_depths.append(self.model.depth_map(source_))
            # 添加相机字条
            cam_poses.append(pose)
            # 添加整个变化矩阵
            trans = transformation_from_parameters(pose[..., :3], pose[..., 3:])
            cam_trans.append(trans)

        # 可是每个深度图
        for s in range(start_scale, self.multi_scale):
            self.ploter.show_image_tensor(cur_images[s] * 255)

            for frame_id in range(len(source_images)):
                _, depth = disp_to_depth(depth_maps[s], self.min_depth, self.max_depth)
                grids, computed_depth = get_sample_grid(depth, Ks[s], inv_Ks[s], cam_trans[frame_id])
                source2target = view_syn(source_images[frame_id][s], grids)
                self.ploter.show_image_tensor(source_images[frame_id][s] * 255)
                self.ploter.show_image_tensor(source2target * 255)
                # 可视化结果
                self.ploter.show_depth(depth)
                if self.using_sc_depth:
                    self.show_diff_map(grids, computed_depth, refer_depths[frame_id][s])

    def update_model(self, model):
        self.model = model

    def show_diff_map(self, grids, computed_map, refer_depth):
        _, refer_depth = disp_to_depth(refer_depth, self.min_depth, self.max_depth)
        syn_depth = view_syn(refer_depth, grids)
        diff_depth = (computed_map - syn_depth).abs() / (computed_map + syn_depth)
        weight_mask = (1 - diff_depth)
        print(torch.max(weight_mask), torch.min(weight_mask))
        self.ploter.show_depth(refer_depth)
        self.ploter.show_depth(weight_mask)
