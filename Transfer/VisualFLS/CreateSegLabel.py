# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 9:54
# @Author  : ljq
# @desc    : 
# @File    : CreateSegLabel.py
import os
import json
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_crop(mask_img, crop_size):
    # crop_size: [x,y,w,h]
    x, y, w, h = crop_size
    mask_img = mask_img[y:y + h, x:x + w]
    return mask_img


def drawing_mask(img_shape, mask_polygons):
    mask_img = np.zeros(img_shape, np.uint8)
    # 绘制mask
    for i, polygons in enumerate(mask_polygons):
        for pts in polygons:
            cv2.fillPoly(mask_img, [np.array(pts, np.int32).reshape(-1, 2)], i + 1)
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


def create_seg_mask(root_path, img_base='imgs', label_str=['LockHole']):
    crop_size = [420, 0, 1080, 1080]
    seg_root = os.path.join(root_path, 'seg')
    img_root = os.path.join(root_path, img_base)
    if not os.path.exists(seg_root):
        os.makedirs(seg_root)
    img_files = [f for f in os.listdir(img_root) if f.endswith('jpg')]
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_root, img_file), cv2.IMREAD_COLOR)
        label_file = os.path.join(img_root, img_file.replace('jpg', 'json'))
        if os.path.exists(label_file):
            mask_polygons = get_mask_polygons(label_file, label_str)
        else:
            mask_polygons = []
        mask_img = drawing_mask(img.shape[:2], mask_polygons)
        mask_img = img_crop(mask_img, crop_size)
        cv2.imwrite(os.path.join(seg_root, img_file.replace('jpg', 'png')), mask_img)
    time.sleep(1)


if __name__ == '__main__':
    create_seg_mask('/backup/VisualFLS/17-32-01')
    # label_str = ['LockHole']
    # dataset_root = '/backup/VisualFLS/17-32-01'
    # crop_size = [420, 0, 1080, 1080]
    # seg_root = os.path.join(dataset_root, 'seg')
    # img_root = os.path.join(dataset_root, 'imgs')
    # if not os.path.exists(seg_root):
    #     os.makedirs(seg_root)
    # img_files = [f for f in os.listdir(img_root) if f.endswith('jpg')]
    # # 绘制segmentation
    # for img_file in img_files:
    #     img = cv2.imread(os.path.join(img_root, img_file), cv2.IMREAD_COLOR)
    #     label_file = os.path.join(img_root, img_file.replace('jpg', 'json'))
    #     mask_polygons = get_mask_polygons(label_file, label_str)
    #     mask_img = drawing_mask(img.shape[:2], mask_polygons)
    #     mask_img = img_crop(mask_img, crop_size)
    #     cv2.imwrite(os.path.join(seg_root, img_file), mask_img)
