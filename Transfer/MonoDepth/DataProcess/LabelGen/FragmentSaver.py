# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 11:41
# @Author  : ljq
# @desc    : 
# @File    : FragmentSaver.py
import os
import cv2


class MonoDepthVideoFramgment():
    def __init__(self, save_root) -> None:
        # 需要定义好
        self.save_root = save_root
        self.current_fragment = 0
        self.current_subdir = "image_{}".format(self.current_fragment)
        self.current_frame_id = 0
        self.save_prefix = os.path.join(self.save_root, self.current_subdir)


    def new_fragment(self, current_frame=None):
        if current_frame is None:
            self.current_fragment += 1
        else:
            self.current_fragment = current_frame
        self.current_subdir = "image_{}".format(self.current_fragment)
        self.current_frame_id = 0
        self.save_prefix = os.path.join(self.save_root, self.current_subdir)
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

    def write(self, frame):
        # 保存图片
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)
        img_name = "{}.png".format(self.current_frame_id)
        self.current_frame_id += 1
        cv2.imwrite(os.path.join(self.save_prefix, img_name), frame)
