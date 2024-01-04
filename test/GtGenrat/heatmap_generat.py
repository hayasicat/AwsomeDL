# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 16:07
# @Author  : ljq
# @desc    : 
# @File    : heatmap_generat.py

import numpy as np
import matplotlib.pyplot as plt

from ThirdPack.CuUnet.HumanPts import pts2heatmap


def gaussian_matrix(time_step=5, sigma=1.0):
    mu = 0
    x, y = np.meshgrid(np.linspace(-1, 1, time_step), np.linspace(-1, 1, time_step))
    exp_value = np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * (sigma ** 2)))
    # 权重值
    return exp_value


def draw_heat_map(pts, heatmap_shape, time_step=10, sigma=1.0):
    heatmap = np.zeros((pts.shape[0], heatmap_shape[0], heatmap_shape[1]))


def draw_gaussian_before(sigma):
    # Draw a 2D gaussian
    tmp_size = np.ceil(3 * sigma)

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    #  提高提高方差
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (tmp_size ** 2))
    return g


if __name__ == "__main__":
    empty_img = np.zeros((300, 300), np.uint8)
    pts = np.array([[150, 150]])
    result = gaussian_matrix(time_step=21, sigma=0.2)
    print(result[15, :])
    print(np.max(result))
    plt.imshow(result)
    plt.show()

    origin_result = draw_gaussian_before(3)
    print(origin_result[3, :])
    print(np.max(origin_result))
    plt.imshow(origin_result)
    plt.show()
