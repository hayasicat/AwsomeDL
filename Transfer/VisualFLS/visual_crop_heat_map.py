# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 19:09
# @Author  : ljq
# @desc    : 
# @File    : visual_crop_heat_map.py
import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from ThirdPack.CuUnet.HumanPts import pts2heatmap

img_root = r'/backup/VisualFLS/crop_imgs'
js_root = r'/backup/VisualFLS/seg'
files = os.listdir(js_root)
js_files = [f_name for f_name in files if f_name.endswith('.json')]
kp_types = ['HoistedContainerCorner', 'ContainerSurfaceCorner']

transform = albu.Compose(
    [albu.IAAPerspective(p=0.5)],
    keypoint_params=albu.KeypointParams(format='xy')
)

for js_name in js_files:
    current_kp = json.loads(open(os.path.join(js_root, js_name), 'r', encoding='utf-8').read())
    img = cv2.imread(os.path.join(img_root, js_name.replace('.json', '.jpg')), cv2.IMREAD_COLOR)
    mask = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
    pt_list = []
    for idx, corner_name in enumerate(kp_types):
        if corner_name not in list(current_kp.keys()):
            continue
        pt_list.append(current_kp[corner_name][0])
    #     pts = np.array(current_kp[corner_name]).reshape((-1, 2))
    #     heat_map, _ = pts2heatmap(pts, img.shape, sigma=10)
    #     mask[:, :, idx] = (heat_map[0, :, :] * 255).astype(np.uint8)
    # dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    # plt.imshow(dst)
    # plt.show()

    result = transform(image=img, keypoints=pt_list)
    img = result['image']
    pt_list = result['keypoints']
    for idx, pt in enumerate(pt_list):
        pts = np.array(pt).reshape((-1, 2))
        heat_map, _ = pts2heatmap(pts, img.shape, sigma=10)
        mask[:, :, idx] = (heat_map[0, :, :] * 255).astype(np.uint8)
    dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    plt.imshow(dst)
    plt.show()
