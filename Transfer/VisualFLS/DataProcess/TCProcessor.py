# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 16:26
# @Author  : ljq
# @desc    : 
# @File    : TCProcessor.py
import copy

from Transfer.VisualFLS.DataProcess.base import parser_json, drawing_mask, img_crop, BaseProcessor
from Transfer.VisualFLS.DataProcess.YHProcessor import YHProcessor


class TCProcessor(YHProcessor):
    grasp_container = {
        'LU': [1295, 381],
        'LB': [1303, 33],
        'RU': [821, 441],
        'RB': [1019, 15],
        'rect_shape': 1024
    }

    fold_container = {
        'LU': [687, 680],
        'LB': [664, 682],
        'RU': [1525, 405],
        'RB': [1284, 435],
        'rect_shape': 512
    }

    def __init__(self):
        super().__init__()

    def get_shape(self, task_id, img_path, js_dict, img_shape):
        """
        task id 1为抓箱子，2为叠箱子
        :param task_id:
        :return:
        """
        channel_name = img_path.rsplit('.', 1)[0].split('_')[-1]
        if task_id == 1:
            init_dict = self.grasp_container
            base_point = js_dict["ContainerSurfaceCorner"][0][0]
        else:
            init_dict = self.fold_container
            base_point = js_dict['HoistedContainerCorner'][0][0]
        init_p = copy.deepcopy(init_dict[channel_name])
        init_p.extend([init_dict['rect_shape'], init_dict['rect_shape']])
        crop_point = [int(base_point[0]) - init_dict['rect_shape'] // 2,
                      int(base_point[1]) - init_dict['rect_shape'] // 2, init_dict['rect_shape'],
                      init_dict['rect_shape']]
        if crop_point[0] + crop_point[2] > img_shape[1]:
            crop_point[0] = img_shape[1] - crop_point[2]
        if crop_point[1] + crop_point[3] > img_shape[0]:
            crop_point[1] = img_shape[0] - crop_point[3]
        return crop_point
