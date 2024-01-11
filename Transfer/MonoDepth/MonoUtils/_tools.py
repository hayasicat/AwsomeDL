# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 9:55
# @Author  : ljq
# @desc    : 
# @File    : _tools.py

def disp_to_depth(disp, min_depth, max_depth):
    """
    monodepth2
    :param disp:
    :param min_depth:
    :param max_depth:
    :return:
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth