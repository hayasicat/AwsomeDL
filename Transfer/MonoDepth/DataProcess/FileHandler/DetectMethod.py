# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 9:38
# @Author  : ljq
# @desc    : 
# @File    : DetectMethod.py
import cv2


class DiffFrameDetector():
    def __init__(self, frame_num=2) -> None:
        self.threshold = 15
        self.previous_frame = None
        # 用0和1来代替
        self.previous_state = 0
        self.current_state = 0
        self.miss_activate_frame = 10 * 4
        self.current_miss_frame = 0

    def is_activate(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        gray_frame = cv2.equalizeHist(gray_frame)

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return None
        # 差帧法找活动的区间
        diff_frame = cv2.absdiff(self.previous_frame, gray_frame)
        ret, diff_frame = cv2.threshold(diff_frame, self.threshold, 255, cv2.THRESH_BINARY)

        count = cv2.countNonZero(diff_frame)
        # shower.imshow("diff",diff_frame)
        # shower.imshow("previous",self.previous_frame)
        # shower.imshow("current",gray_frame)
        # shower.imshow("diff",diff_frame)
        # cv2.waitKey(1)
        self.previous_frame = gray_frame
        self.previous_state = self.current_state
        activate_region = count / (gray_frame.shape[0] * gray_frame.shape[1])
        if activate_region > 0.05:
            self.current_state = 1
            self.current_miss_frame = 0
        elif self.current_state == 1 and self.current_miss_frame < self.miss_activate_frame:
            self.current_miss_frame += 1
        else:
            self.current_state = 0
        # 为了容错性，检测到在动之后要连续三帧才能转变
        return True
