# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 11:32
# @Author  : ljq
# @desc    : 
# @File    : CVIMSHOWER.py
import cv2


class ImshowWindows():
    def __init__(self) -> None:
        self.namewindows_dict = {}

    def imshow(self, windows_name, img):
        if windows_name not in list(self.namewindows_dict.keys()):
            self.namewindows_dict[windows_name] = cv2.namedWindow(windows_name)
        cv2.imshow(windows_name, img)

    def fresh(self, wait_times=10):
        # 判定案件输入
        c = cv2.waitKey(wait_times)
        return c

    def destory_all(self):
        cv2.destroyAllWindows()
