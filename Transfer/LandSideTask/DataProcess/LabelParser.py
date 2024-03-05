# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 15:40
# @Author  : ljq
# @desc    : 这边是将labelme转换为yolo得样子
# @File    : labelme2yolo.py
import os
import cv2
import json


class LabelMeFormatParser:
    def __init__(self):
        """
        data_root:
            - imgs
            - labels
            - index.txt # 类别1,类别2，类别3
        :param dataset_root:
        """
        pass

    @staticmethod
    def parse(img_path):
        # 解析分为集中，一种BoundBox
        anns = []
        img_root, img_name = os.path.split(img_path)
        suffix = img_name.rsplit('.')[-1]
        js_path = img_path.replace('.' + suffix, '.json')
        js_ann = json.loads(open(js_path, 'r', encoding='utf-8').read())
        shapes = js_ann['shapes']
        for s in shapes:
            l = s['label']
            s_type = s['shape_type']
            point_list = s['points']
            # 归一化同一个标签，如果是一个点的话就是关键点，如果是两个的话可能是BBOX
            # TODO: points -> 还需要定位到在哪个物体里面
            # TODO: bbox的转换
            current_ann = {}
            current_ann[l] = point_list
            anns.append(current_ann)
        return js_path, img_path, anns


if __name__ == "__main__":
    p = LabelMeFormatParser(r'G:\FLSNEW\example\2024-02-23')
