# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 15:54
# @Author  : ljq
# @desc    : 
# @File    : MonoViewer.py
import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class MonoViewer():
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
