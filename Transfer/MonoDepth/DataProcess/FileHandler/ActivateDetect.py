# -*- coding: utf-8 -*-
# @Time    : 2023/9/19 9:27
# @Author  : ljq
# @desc    : 
# @File    : ActivateDetect.py
from .VideoObject import CV2VideoObject
from .JsonFileParser import JosnParser
from .DetectMethod import DiffFrameDetector


class VideoActivateDetector():
    # TODO： 增加不同的检测方法
    def __init__(self, detect_method=None) -> None:
        self.detector = DiffFrameDetector()

    def open(self, video_path):
        return CV2VideoObject(video_path)

    def detect(self, video_stream: CV2VideoObject):
        activate_interval = []
        fragment = {}
        while True:
            frame = video_stream.get_frame()
            # 测试一下
            if frame is None:
                break
            self.detector.is_activate(frame)
            if self.detector.previous_state == 0 and self.detector.current_state == 1:
                fragment["start_time"] = video_stream.timestamp()
                fragment["start_frame"] = video_stream.cframe()
            if self.detector.previous_state == 1 and self.detector.current_state == 0:
                fragment["end_time"] = video_stream.timestamp()
                fragment["end_frame"] = video_stream.cframe()
                activate_interval.append(fragment)
                fragment = {}
        return activate_interval

    def flush(self, activate_interval, file_path):
        # 将时间的区间数据写入到硬盘中
        filehandler = JosnParser(file_path)
        filehandler.write(activate_interval)
