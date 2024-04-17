# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 9:38
# @Author  : ljq
# @desc    : 
# @File    : DetectMethod.py
import cv2

import torch
import torch.nn as nn

import numpy as np


class DiffFrameDetector():
    def __init__(self, frame_num=2) -> None:
        self.threshold = 15
        self.previous_frame = None
        # 用0和1来代替
        self.previous_state = 0
        self.current_state = 0
        self.miss_activate_frame = 10 * 4
        self.current_miss_frame = 0

    def is_activate(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        gray_frame = cv2.equalizeHist(gray_frame)

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return None
        # 差帧法找活动的区间
        diff_frame = cv2.absdiff(self.previous_frame, gray_frame)
        ret, diff_frame = cv2.threshold(diff_frame, self.threshold, 255, cv2.THRESH_BINARY)

        count = cv2.countNonZero(diff_frame)
        # shower.imshow("diff",diff_frame)
        # shower.imshow("previous",self.previous_frame)
        # shower.imshow("current",gray_frame)
        # shower.imshow("diff",diff_frame)
        # cv2.waitKey(1)
        self.previous_frame = gray_frame
        self.previous_state = self.current_state
        activate_region = count / (gray_frame.shape[0] * gray_frame.shape[1])
        if activate_region > 0.05:
            self.current_state = 1
            self.current_miss_frame = 0
        elif self.current_state == 1 and self.current_miss_frame < self.miss_activate_frame:
            self.current_miss_frame += 1
        else:
            self.current_state = 0
        # 为了容错性，检测到在动之后要连续三帧才能转变
        return True


class SSIM(nn.Module):
    # https://github.com/JiawangBian/sc_depth_pl/blob/master/losses/loss_functions.py
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self, kernel_size=7):
        super(SSIM, self).__init__()
        k = kernel_size
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k // 2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
                 (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class MoveDetector():
    def __init__(self, threshold):
        self.ssim = SSIM(7)
        self.cache_list = []
        self.threshold = threshold

    def is_activate(self, frame):
        frame = cv2.resize(frame, (896, 416))
        # 把frame 归一化，然后取片段
        frame = frame.astype(np.float32) / 255.0
        # 转为tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        if len(self.cache_list) == 0:
            self.cache_list.append(frame_tensor)
            return True
        history_frame = self.cache_list[0]
        noisy = 0.85 * self.ssim(frame_tensor, history_frame)[0] + 0.15 * torch.abs(
            frame_tensor - history_frame)[0]
        noisy = noisy.mean([0, 1])
        print(noisy)
        # noisy超过一定阈值以后
        if noisy >= self.threshold:
            self.cache_list.pop(0)
            self.cache_list.append(frame_tensor)
            return True
        return False

    def clear(self):
        self.cache_list = []


class SCDepthDetector():
    def __init__(self, threshold=0.5):
        self.cache_list = []
        self.threshold = threshold

    def is_activate(self, frame):
        frame = cv2.resize(frame, (896, 416))
        frame = self.crop(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 来个高斯平滑，不然箱面区域的采样实在是太过于密集了
        frame_gray = cv2.medianBlur(frame_gray, 5)

        if len(self.cache_list) == 0:
            self.cache_list.append(frame_gray)
            return True
        history_frame = self.cache_list[0]
        h, w = frame_gray.shape
        diff = np.abs(history_frame - frame_gray)
        # 因为纹理比较多如果只是>10那就是走大车比较短采样小车比价长采样
        ratio = (diff > 10).sum() / (h * w)
        # 连续超过三次阈值以后才开始保存
        if ratio >= self.threshold:
            # cv2.imshow("history_img", history_frame)
            # cv2.imshow("cur", frame_gray)
            # cv2.imshow("img", (diff > 15).astype(np.uint8) * 255)
            # cv2.waitKey(0)
            self.cache_list.pop(0)
            self.cache_list.append(frame_gray)
            return True
        return False

    def clear(self):
        self.cache_list = []

    def crop(self, img):
        return img[:300, :, :]


class FeatureMatchDetector():
    def __init__(self, threshold=10):
        # 虽然比较慢一点，按时能得到一个稳定结果的话，也还算是还不错

        pass
