# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 14:57
# @Author  : ljq
# @desc    : 
# @File    : base.py
import json

import cv2
import numpy as np


def drawing_mask(img_shape, mask_polygons):
    mask_img = np.zeros(img_shape, np.uint8)
    # 绘制mask
    for i, polygons in enumerate(mask_polygons):
        for pts in polygons:
            cv2.fillPoly(mask_img, [np.array(pts, np.int32).reshape(-1, 2)], i + 1)
    return mask_img


def img_crop(mask_img, crop_size):
    # crop_size: [x,y,w,h]
    x, y, w, h = crop_size
    mask_img = mask_img[y:y + h, x:x + w]
    return mask_img


def get_mask_polygons(json_label, label_str):
    label = json.loads(open(json_label, 'r', encoding='utf-8').read())
    # 解析一下
    shapes = label['shapes']
    # 分别读取几个标签
    mask_polygons = []
    for tag in label_str:
        polygons = []
        for sp in shapes:
            if sp['label'] == tag:
                polygons.append(sp['points'])
        mask_polygons.append(polygons)
    return mask_polygons


def parser_json(json_label):
    """
    解析json数据
    :param json_label:
    :return:
    """
    json_dict = {}
    label = json.loads(open(json_label, 'r', encoding='utf-8').read())
    # 解析一下
    shapes = label['shapes']
    # 分别读取几个标签
    mask_polygons = []
    for shape in shapes:
        label_name = shape['label']
        mask_polygons = json_dict.get(label_name, [])
        mask_polygons.append(shape['points'])
        json_dict[label_name] = mask_polygons
    return json_dict


class BaseProcessor():
    mask_label_str = ['LockHole', 'HoistedContainer']
    kp_types = ['HoistedContainerCorner', 'ContainerSurfaceCorner']
