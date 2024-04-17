# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 14:49
# @Author  : ljq
# @desc    : 
# @File    : YHProcessor.py
import os
import copy

import cv2

from Transfer.VisualFLS.DataProcess.base import parser_json, drawing_mask, img_crop, BaseProcessor
from .coordinates_overlap import get_overlap_coordinates


class YHProcessor(BaseProcessor):
    # 就是一些标签

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

    def __init__(self):
        super().__init__()

    def transform(self, img_path):
        # 返回需要保存的信息
        js_path = img_path.replace('.jpg', '.json')
        js_dict = parser_json(js_path)
        # 制造一个mask的标签
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_polygons = [js_dict.get(self.mask_label_str[0], []), js_dict.get(self.mask_label_str[1], [])]
        mask_img = drawing_mask(img.shape[:2], mask_polygons)
        # crop 原始图片以及mask 图片
        task_id = 1
        if len(mask_polygons[-1]) > 0:
            task_id = 2
        crop_size = self.get_shape(task_id, img_path, js_dict, img.shape)
        mask_img = img_crop(mask_img, crop_size)
        crop_img = img_crop(img, crop_size)
        result = {'img_patch': crop_img, 'mask_patch': mask_img}
        # 关键点坐标的生成
        result['corner_points'] = self.get_corner_points(js_dict, crop_size)
        # crop_coordinates 这边用来生成相应的标签
        crop_polygons = self.crop_coordinates(crop_size, mask_polygons)
        result['crop_polygons'] = crop_polygons
        return result

    def crop_coordinates(self, crop_size, mask_polygons):
        """裁剪图片"""
        x, y, w, h = crop_size
        crop_coor = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        crop_polygons = []
        for polygons in mask_polygons:
            crop_polygon = []
            for p in polygons:
                is_intersection, inter_section_points = get_overlap_coordinates(crop_coor, p)
                if is_intersection:
                    format_inter_section_points = self.format_polygons(crop_size, inter_section_points)
                    crop_polygon.extend(format_inter_section_points)
            crop_polygons.append(crop_polygon)
        # 格式化坐标点
        return crop_polygons

    def format_polygons(self, crop_size, polygons):
        x, y, w, h = crop_size
        formated_polygons = []
        for p in polygons:
            new_p = []
            for x1, y1 in p:
                new_p.append([(x1 - x) / w, (y1 - y) / h])
            formated_polygons.append(new_p)
        return formated_polygons

    def get_corner_points(self, json_dict, crop_size):
        corner_points_dict = {}
        for point_type in self.kp_types:
            if len(json_dict.get(point_type, [])) > 0:
                corner_points_dict[point_type] = json_dict[point_type][0]
        # 重新对所有的店进行裁剪
        for type_name, type_value in corner_points_dict.items():
            for idx, num in enumerate(crop_size[:2]):
                type_value[0][idx] -= num
        return corner_points_dict

    def get_shape(self, task_id, img_path, js_dict, img_shape):
        """
        task id 1为抓箱子，2为叠箱子
        :param task_id:
        :return:
        """
        channel_name = img_path.split('_')[-2]
        if task_id == 1:
            init_dict = self.grasp_container
        else:
            init_dict = self.fold_container
        init_p = copy.deepcopy(init_dict[channel_name])
        init_p.extend([init_dict['rect_shape'], init_dict['rect_shape']])
        return init_p


if __name__ == "__main__":
    print(YHProcessor().transform(r'/root/data/VisualFLS/imgs/2023-10-02_18-33-19_c2_2.jpg'))
