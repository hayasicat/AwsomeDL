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


def visualTensor(image_tensor):
    # batchsize就一第一张
    if len(image_tensor.size()) > 3:
        image_tensor = image_tensor[0, ...]
    new_image = image_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    plt.imshow(new_image)
    plt.show()


def visualReproject(loss_tensor):
    # 重投影的loss可视化
    if len(loss_tensor.size()) > 3:
        loss_tensor = loss_tensor[0, ...]
    loss_tensor = loss_tensor.detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(loss_tensor)
    plt.show()


if __name__ == "__main__":
    from Transfer.MonoDepth.dataset import MonoDataset
    from Transfer.MonoDepth.Losses.Loss import ReprojectLoss, EdgeSmoothLoss, AutoMask

    data_root = r'/root/project/AwsomeDL/data/BowlingMono'
    train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
    # train_data = MonoDataset(data_root, train_file_path, 416, 896)
    train_data = MonoDataset(data_root, train_file_path, 832, 1824)
    from torch.utils.data import DataLoader
    from Transfer.MonoDepth.MonoUtils.CameraTrans import get_sample_grid, transformation_from_parameters

    g_loss = ReprojectLoss()
    train_loader = DataLoader(train_data, 1)
    for idx, inputs in enumerate(train_loader):
        if idx <= 12:
            continue
        image = inputs['prime0_3']
        image_next = inputs['prime1_3']
        image_pre = inputs['prime-1_3']
        K = inputs['K_3']
        inv_K = inputs['inv_K3']
        auto_mask = AutoMask()
        mask = auto_mask.compute_real_project(image, image_next, image_pre)
        fake_depth = torch.ones((image.size(0), 1, image.size(2), image.size(3))) * 5
        # 伪造一个旋转矩阵，和一个转移矩阵
        rot = torch.zeros((1, 1, 3))
        trans = torch.zeros((1, 1, 3))
        trans[:, :, 0] = 0.025 * 10
        #
        T = transformation_from_parameters(rot, trans)
        print(T)
        sample_grid = get_sample_grid(fake_depth, K, inv_K, T)
        # 合成一个新视角得图片
        new_image = view_syn(image_next, sample_grid)
        visualTensor(image * 255)
        visualTensor(image_next * 255)
        visualTensor(new_image * 255)
        source_loss = g_loss(image, image_next)
        reproject_loss = g_loss(image, new_image)

        visualReproject(source_loss)
        visualReproject(reproject_loss)

        # automask loss
        automask_loss = auto_mask(image, image_next, mask)
        visualReproject(automask_loss)
        print("source:", source_loss.mean())
        print("reproject:", reproject_loss.mean())
        print("automask:", automask_loss.mean())
        break
