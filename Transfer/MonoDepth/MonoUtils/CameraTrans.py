# -*- coding: utf-8 -*-
# @Time    : 2023/12/28 10:35
# @Author  : ljq
# @desc    : 
# @File    : CameraTrans.py
import torch
import torch.nn as nn
import torchvision
import numpy as np


def get_sample_grid(depth, K, inv_K, T):
    """
    相机坐标反向投影回世界坐标
    :param depth:
    :return:
    """
    b, c, h, w = depth.size()
    Y, X = torch.meshgrid(torch.arange(0, float(depth.size(2))),
                          torch.arange(0, float(depth.size(3))))  # 输出的是Y,X 排序一下
    pixel_coord = torch.stack([X, Y, torch.ones_like(X)]).view(3, -1).to(K.device)

    cam_points = torch.matmul(inv_K[:, :3, :3], pixel_coord)  # 4X4需要去掉最后一个维度
    ones_vec = torch.ones_like(depth).view(depth.size(0), 1, -1).to(K.device)
    cam_points = depth.view(depth.size(0), 1, -1) * cam_points
    world_points = torch.cat([cam_points, ones_vec], 1)

    # 世界坐标系转换为相机坐标系
    P = torch.matmul(K, T)[:, :3, :]

    cam_points_t = torch.matmul(P, world_points)
    # 归一化
    pix_coords_t = cam_points_t[:, :2, :] / (cam_points_t[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords_t = pix_coords_t.view(b, 2, h, w)
    pix_coords_t = pix_coords_t.permute(0, 2, 3, 1)
    # 因为torch.nn.function.grid_sample 输入时[-1,1],这一步时在归一化
    pix_coords_t[..., 0] /= w - 1
    pix_coords_t[..., 1] /= h - 1
    pix_coords = (pix_coords_t - 0.5) * 2
    # 变换后相机坐标
    return pix_coords, cam_points_t[:, 2, :].view(b, 1, h, w)


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot
