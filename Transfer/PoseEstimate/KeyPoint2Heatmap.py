# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 18:11
# @Author  : ljq
# @desc    : 
# @File    : KeyPoint2Heatmap.py
import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ThirdPack.CuUnet.HumanPts import pts2heatmap

# 读取数据集
root_path = r'/backup/VisualFLS/imgs'
# 筛查关键点
files = os.listdir(root_path)
js_files = [f_name for f_name in files if f_name.endswith('.json')]
kp_types = ['HoistedContainerCorner', 'ContainerSurfaceCorner']
for js_name in js_files:
    current_kp = {}
    anns = json.loads(open(os.path.join(root_path, js_name), 'r', encoding='utf-8').read())['shapes']
    for ann in anns:
        if ann['label'] in kp_types:
            current_kp[ann['label']] = ann['points']
    if len(list(current_kp.keys())) == 0 or 'ContainerSurfaceCorner' not in list(current_kp.keys()):
        # 关键点要确定是不是在图片上，之后生成的heatmap直接regression
        continue
    # 读取图片
    img = cv2.imread(os.path.join(root_path, js_name.replace('.json', '.jpg')), cv2.IMREAD_COLOR)
    mask = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
    for idx, corner_name in enumerate(kp_types):
        if corner_name not in list(current_kp.keys()):
            continue
        pts = np.array(current_kp[corner_name]).reshape((-1, 2))
        heat_map, _ = pts2heatmap(pts, img.shape, sigma=10)
        mask[:, :, idx] = (heat_map[0, :, :] * 255).astype(np.uint8)
    print(current_kp)
    dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    plt.imshow(dst)
    plt.show()
    break
