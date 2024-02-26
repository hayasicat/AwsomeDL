# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 17:18
# @Author  : ljq
# @desc    : 
# @File    : rename.py
import os
from shutil import copyfile


def add_prefix_tag(project_location, img_root):
    img_files = os.listdir(img_root)
    for img_name in img_files:
        copyfile(os.path.join(img_root, img_name), os.path.join(img_root, project_location + '_' + img_name))
        os.remove(os.path.join(img_root, img_name))


if __name__ == "__main__":
    add_prefix_tag('tc', r'/root/data/VisualFLS/newly')
