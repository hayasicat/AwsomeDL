# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 11:37
# @Author  : ljq
# @desc    : 
# @File    : VideoObject.py
import cv2


class CV2VideoObject():
    """
    TODO: 添加进度条跳转，方便多进程处理
    """

    def __init__(self, video_path) -> None:
        self.video_stream = cv2.VideoCapture(video_path)

    def close(self):
        self.video_stream.relase()

    def get_frame(self):
        success, frame = self.video_stream.read()
        if not success:
            print("read frame failure")
        return frame

    def timestamp(self):
        cframe = self.video_stream.get(cv2.CAP_PROP_POS_FRAMES)  # retrieves the current frame number
        # tframe = self.video_stream.get(cv2.CV_CAP_PROP_FRAME_COUNT) # get total frame count
        fps = self.video_stream.get(cv2.CAP_PROP_FPS)  # get the FPS of the videos
        time = cframe / fps
        return time

    def cframe(self):
        return self.video_stream.get(cv2.CAP_PROP_POS_FRAMES)

    def locate2frame(self, frame_location):
        """
        跳转到特定帧数
        """
        self.video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_location)

    def skip(self, frame_num):
        for i in range(frame_num):
            self.get_frame()
