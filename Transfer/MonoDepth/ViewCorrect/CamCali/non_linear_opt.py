# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 11:13
# @Author  : ljq
# @desc    : 
# @File    : non_linear_opt.py
import torch

import numpy as np
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR

# 这么小的值直接给个CPU就可以了
device = torch.device("cpu")


class CamModel(nn.Module):
    def __init__(self, k, r, t):
        super(CamModel, self).__init__()
        # 创建一个相机模型
        self.t = nn.Parameter(torch.from_numpy(t), requires_grad=True)
        self.r = nn.Parameter(torch.from_numpy(r), requires_grad=True)
        # self.k = nn.Parameter(torch.ones((3, 3)), requires_grad=True)
        # 这几个值才是必须的，不给大的K矩阵，就给一个约束的结果
        self.fx = nn.Parameter(torch.Tensor([k[0, 0]]), requires_grad=True)
        self.fy = nn.Parameter(torch.Tensor([k[1, 1]]), requires_grad=True)
        self.cx = nn.Parameter(torch.Tensor([k[0, 2]]), requires_grad=True)
        self.cy = nn.Parameter(torch.Tensor([k[1, 2]]), requires_grad=True)

    def forward(self, pw):
        # [b,3]
        cam_points = (self.r @ (pw + self.t).permute(0, 2, 1)).squeeze(-1)
        cam_points = torch.stack([cam_points[:, 0] * self.fx + cam_points[:, 2] * self.cx,
                                  cam_points[:, 1] * self.fy + cam_points[:, 2] * self.cy,
                                  cam_points[:, 2]], dim=1)
        pixel_points = cam_points / cam_points[:, 2].view(-1, 1)

        # 转换为pixel的坐标点
        # pixel_point = cam_point / cam_point[:, -1]
        return pixel_points


class CamCalibrationOpt():
    # 非线性校正一下
    def __init__(self, k, r, t):
        self.model = CamModel(k, r, t)
        self.optimizer = optim.Adam([
            {"params": self.model.t, "lr": 0.003},
            {"params": self.model.r, "lr": 0.003},
            {"params": [self.model.fx, self.model.cy, self.model.fy, self.model.cx], "lr": 0.003},
        ])
        self.scheduler_1 = StepLR(self.optimizer, step_size=20, gamma=0.99)

    def opt(self, pw, px):
        for i in range(500):
            self.optimizer.zero_grad()
            cal_px = self.model(pw)
            loss = torch.square(px - cal_px).sum()
            print(loss)
            loss.backward()
            self.optimizer.step()
            self.scheduler_1.step()
        print(self.model.r, self.model.t, self.model.fx, self.model.fy, self.model.cx, self.model.cy)

    def export(self):
        K = np.zeros((3, 3))
        K[0, 0] = self.model.fx.detach().numpy()[0]
        K[0, 2] = self.model.cx.detach().numpy()[0]
        K[1, 1] = self.model.fy.detach().numpy()[0]
        K[1, 2] = self.model.cy.detach().numpy()[0]
        K[2, 2] = 1
        return K, self.model.r.detach().numpy(), self.model.t.detach().numpy()


if __name__ == '__main__':
    # world_points = [
    #     [12192, 2438, -2896, 1], [12192, 0, -2896, 1], [0, 0, -2896, 1], [0, 2438, -2896, 1],
    #     [12192, 2438, -5792, 1], [12192, 0, -5792, 1], [0, 0, -5792, 1], [0, 2438, -5792, 1],
    #     [12192, 2438, -8688, 1], [12192, 0, -8688, 1], [0, 0, -8688, 1], [0, 2438, -8688, 1],
    #     [12192, 2438, -11584, 1], [12192, 0, -11584, 1], [0, 0, -11584, 1], [0, 2438, -11584, 1],
    #     [12192, 2438, -14480, 1], [12192, 0, -14480, 1], [0, 0, -14480, 1], [0, 2438, -14480, 1],
    # ]
    # 
    # image_points = [
    #     [1222, 693], [1223, 575], [640, 557], [634, 676],
    #     [1266, 707], [1266, 570], [588, 551], [581, 688],
    #     [1329, 727], [1330, 562], [520, 539], [509, 702],
    #     [1421, 757], [1421, 554], [414, 525], [399, 727],
    #     [1571, 809], [1565, 538], [241, 497], [217, 766],
    # ]
    world_points = [
        [12192, 2438, -2896, 1], [12192, 0, -2896, 1], [0, 0, -2896, 1], [0, 2438, -2896, 1],
        [12192, 2438, -5792, 1], [12192, 0, -5792, 1], [0, 0, -5792, 1], [0, 2438, -5792, 1],
        [12192, 2438, -8688, 1], [12192, 0, -8688, 1], [0, 0, -8688, 1], [0, 2438, -8688, 1],
        [12192, 2438, -11584, 1], [12192, 0, -11584, 1], [0, 0, -11584, 1], [0, 2438, -11584, 1],

    ]
    image_points = [
        [1395, 646], [1390, 505], [681, 525], [686, 667],
        [1487, 674], [1482, 501], [623, 527], [629, 700],
        [1627, 715], [1621, 496], [537, 530], [546, 745],
        [1882, 781], [1877, 482], [388, 525], [397, 820],
    ]

    world_points = np.array(world_points)
    image_points = np.array(image_points)

    K = np.array([[947.72514102, 5.9731722, 938.91474948],
                  [0., 946.47854876, 536.57768401],
                  [0., 0., 1.]])
    R = np.array(
        [[0.99950494, 0.03071985, 0.00679409],
         [-0.03063415, 0.99945402, - 0.0123777],
         [-0.00717062, 0.01216345, 0.99990031]]
    )
    T = np.array(
        [[-4481.12003764],
         [-185.6196051],
         [19393.99550627]]
    ).reshape((1, 1, -1))

    slover = CamCalibrationOpt(K, R, T)
    world_points = torch.from_numpy(world_points[:, :3].reshape(-1, 1, 3))
    one_vec = np.ones(image_points.shape[0]).reshape((-1, 1))
    image_points = torch.from_numpy(np.hstack([image_points, one_vec]).reshape(-1, 3))
    # print(world_points[:, :3], image_points.shape)
    slover.opt(world_points[4:, :, :], image_points[4:, :])
    print(slover.export())
