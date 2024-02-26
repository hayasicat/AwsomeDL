# -*- coding: utf-8 -*-
# @Time    : 2024/1/31 18:34
# @Author  : ljq
# @desc    : 
# @File    : freme_noisy_calculate.py
import os

import cv2
import numpy as np
from Transfer.MonoDepth.DataProcess.FileHandler.DetectMethod import MoveDetector, SCDepthDetector

detector = SCDepthDetector(0.6)


# 强行切分一下

def compute_movement_ratio(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h, w = frame1_gray.shape
    diff = np.abs(frame1_gray - frame2_gray)
    ratio = (diff > 10).sum() / (h * w)
    return ratio


img_root = '/root/data/BowlingMono/fragments/newnvr238_ch8_20230803000011_20230803105251/image_66/'
filenames = os.listdir(img_root)
filenames = sorted(filenames, key=lambda x: eval(x.split('.')[0]))
# for idx, img_name, next_img_name in zip(range(len(filenames)), filenames, filenames[1:]):
#
#         img0 = cv2.imread(os.path.join(img_root, img_name))
#         img1 = cv2.imread(os.path.join(img_root, next_img_name))
#         # 这个方案不行诶，只能根据行进方向来做一个变化
#         result = compute_movement_ratio(img0, img1)
#         if result < 0.5:
#             print(idx, True)
for idx, img_name in enumerate(filenames):
    img = cv2.imread(os.path.join(img_root, img_name))
    result = detector.is_activate(img)
    print(idx, result)
