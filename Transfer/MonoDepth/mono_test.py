# -*- coding: utf-8 -*-
# @Time    : 2023/12/28 10:38
# @Author  : ljq
# @desc    : 
# @File    : mono_test.py
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def view_syn(source_frame, grid_sample):
    # 视角转换的loss
    source_frame = source_frame.to(grid_sample.dtype)
    new_view = F.grid_sample(
        source_frame,
        grid_sample,
        padding_mode="border")
    return new_view


if __name__ == "__main__":
    from Transfer.MonoDepth.dataset import MonoDataset

    data_root = r'/root/project/AwsomeDL/data/BowlingMono'
    train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
    train_data = MonoDataset(data_root, train_file_path, 832, 1824)
    from torch.utils.data import DataLoader
    from Transfer.MonoDepth.MonoUtils.CameraTrans import Camera2Camera, transformation_from_parameters

    train_loader = DataLoader(train_data, 1)
    for idx, inputs in enumerate(train_loader):
        if idx <= 5:
            continue
        image = inputs['prime0_0']
        image_next = inputs['prime1_0']
        K = inputs['K_0']
        inv_K = inputs['inv_K0']
        fake_depth = torch.ones((image.size(0), 1, image.size(2), image.size(3)))
        # 伪造一个旋转矩阵，和一个转移矩阵
        rot = torch.zeros((1, 1, 3))
        trans = torch.zeros((1, 1, 3))
        trans[:, :, 0] = 0.01

        T = transformation_from_parameters(rot, trans)
        sample_grid = Camera2Camera()(fake_depth, K, inv_K, T)
        # 合成一个新视角得图片
        new_image = view_syn(image_next, sample_grid)
        new_image = new_image.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.show()
        plt.imshow(image_next.squeeze().permute(1, 2, 0))
        plt.show()
        plt.imshow(new_image)
        plt.show()

        break
