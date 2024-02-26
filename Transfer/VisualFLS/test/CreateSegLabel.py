# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 9:54
# @Author  : ljq
# @desc    : 
# @File    : CreateSegLabel.py
import copy
import json
import os
import time

import cv2
import numpy as np

grasp_container = {
    'c1': [503, 310],
    'c2': [455, 310],
    'c3': [446, 10],
    'c4': [497, 2],
    'rect_shape': 768
}

fold_container = {
    'c1': [597, 381],
    'c2': [871, 414],
    'c3': [587, 402],
    'c4': [943, 283],
    'rect_shape': 384
}


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


def create_seg_mask(root_path, img_base='imgs', label_str=['LockHole', 'HoistedContainer'],
                    crop_size=[420, 0, 1080, 1080]):
    seg_root = os.path.join(root_path, 'seg')
    crop_root = os.path.join(root_path, 'crop_imgs')
    img_root = os.path.join(root_path, img_base)
    if not os.path.exists(seg_root):
        os.makedirs(seg_root)
    # 裁剪的固定路径
    if not os.path.exists(crop_root):
        os.makedirs(crop_root)

    img_files = [f for f in os.listdir(img_root) if f.endswith('jpg')]
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_root, img_file), cv2.IMREAD_COLOR)
        label_file = os.path.join(img_root, img_file.replace('jpg', 'json'))
        if os.path.exists(label_file):
            mask_polygons = get_mask_polygons(label_file, label_str)
        else:
            mask_polygons = []
        # 判断是叠箱还是抓箱
        task_id = 1
        if len(mask_polygons[-1]) > 0:
            task_id = 2
        crop_size = get_shape(task_id, img_file.split('_')[-2])
        mask_img = drawing_mask(img.shape[:2], mask_polygons)
        mask_img = img_crop(mask_img, crop_size)

        crop_img = img_crop(img, crop_size)
        cv2.imwrite(os.path.join(seg_root, img_file.replace('jpg', 'png')), mask_img)
        cv2.imwrite(os.path.join(crop_root, img_file), crop_img)

    time.sleep(1)


def get_shape(task_id, channel_name):
    """
    task id 1为抓箱子，2为叠箱子
    :param task_id:
    :return:
    """
    if task_id == 1:
        init_dict = grasp_container
    else:
        init_dict = fold_container
    init_p = copy.deepcopy(init_dict[channel_name])
    init_p.extend([init_dict['rect_shape'], init_dict['rect_shape']])
    return init_p


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
