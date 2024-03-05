# -*- coding: utf-8 -*-
# @Time    : 2024/2/28 11:53
# @Author  : ljq
# @desc    : 
# @File    : YHProcessor.py
from .LabelParser import LabelMeFormatParser
from .LabelConverter import YOLOConverter


class YHProcessor():
    def __init__(self, cls_index):
        self.cls_index = cls_index
        self.parser = LabelMeFormatParser()
        self.converter = YOLOConverter(self.cls_index)

    def transform(self, img_path):
        js_path, img_path, anns = self.parser.parse(img_path)
        # 转换
        content = self.converter.transform(js_path, img_path, anns)
        return content
