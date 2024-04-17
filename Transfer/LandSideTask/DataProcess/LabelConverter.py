# -*- coding: utf-8 -*-
# @Time    : 2024/2/28 10:53
# @Author  : ljq
# @desc    : 
# @File    : LabelConverter.py
import os
import cv2
import numpy as np


class YOLOConverter():
    def __init__(self, cls_list=list()):
        # 查询类别
        self.cls_list = cls_list

    def transform(self, js_path, img_path, anns, **kwargs):
        # 如果是多边形框框的话
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, c = img.shape
        # 需要写入的数据条数和数据内容
        ann_formats = []
        for ann in anns:
            cls_name = list(ann.keys())[0]
            # 现在默认是多边形标注
            points = ann[cls_name]
            cls_name = cls_name.lower()
            if cls_name not in self.cls_list:
                continue
            # 这边要设置为大小写不敏感
            cls_index = self.cls_list.index(cls_name)
            # 转换坐标
            normal_points = [[pts[0] / w, pts[1] / h] for pts in points]
            ann_sample = [cls_index]
            for normal_pts in normal_points:
                ann_sample.extend(normal_pts)
            ann_str_list = [str(i) for i in ann_sample]
            ann_str = ' '.join(ann_str_list)
            ann_formats.append(ann_str)

        label_content = '\n'.join(ann_formats)
        return label_content
