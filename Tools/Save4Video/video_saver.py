import os
import time

import gc
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2


class VideoObserver():
    """
    只保存接口
    """
    alive = False
    cam_name = ''

    def update(self, frame):
        pass

    def add_cache_img(self, img_list):
        pass


class VideoSaver(VideoObserver):
    format_code = {'avi': 'XVID', 'mp4': 'mp4v'}

    def __init__(self, path, cam_name, fps=25, second=10, video_size=[0, 0], video_format='mp4', keep_alive=False,
                 **kwargs):
        """
        video_size: 不指定的话以第一帧为准
        """
        self.syslogger = logging.getLogger('syslog')
        super().__init__()
        self.frame_q = []
        self.video_path = path
        self.cam_name = cam_name
        self.fps = fps
        self.during_time = second
        self.video_size = video_size
        self.video_format = self.format_code[video_format.lower()]
        self.fourcc = cv2.VideoWriter_fourcc(*self.video_format)
        self.total_frame_num = self.fps * self.during_time
        self.video_out = None
        self.alive = True
        self.keep_alive = keep_alive
        self.current_frame = 0
        self.is_writting = False

    def update(self, frame):
        try:
            self.frame_q.append(frame)
            self.record2disk()
        except:
            pass

    def reset(self, path, cam_name):
        """
        重置当前状态
        """
        try:
            # 防止缓存泄露
            del self.frame_q
            gc.collect()
        except:
            pass
        self.frame_q = []
        self.video_path = path
        self.cam_name = cam_name
        self.video_out = None
        self.alive = True
        self.current_frame = 0

    def record2disk(self):
        # 对io进行枷锁
        if self.is_writting:
            return 0
        try:
            self.is_writting = True
            for i in range(len(self.frame_q)):
                frame = self.frame_q.pop(0)
                if self.video_size[0] <= 0 and self.video_size[1] <= 0:
                    h, w = frame.shape[:2]
                    self.video_size = (int(w), int(h))
                if type(self.video_out) == type(None):
                    self.video_out = cv2.VideoWriter(self.video_path, self.fourcc, self.fps,
                                                     self.video_size)
                self.video_out.write(frame)
                self.current_frame += 1
                if self.current_frame >= self.total_frame_num:
                    break
        except Exception as e:
            traceback.print_exc()
        self.is_writting = False

    def stop_record(self):
        self.alive = False
        self.video_out.release()


class FrameSaver(VideoObserver):
    # 只存储单帧图片
    def __init__(self, path, cam_name, time_interval=1):
        super().__init__()
        self.last_time = -1
        self.frame_num = 0
        self.time_interval = time_interval
        self.frame_q = []
        # 保存的地址
        self.path = path
        self.cam_name = cam_name
        self.alive = True
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def update(self, frame):
        # 通过一定的时间频率来保存
        if time.time() - self.last_time > self.time_interval:
            self.frame_q.append(frame)
            self.last_time = time.time()
            self.record2disk()

    def reset(self, path, cam_name):
        # 重置一下保存的方式
        self.path = path
        self.cam_name = cam_name

    def stop_record(self):
        self.alive = False

    def record2disk(self):
        for i in range(len(self.frame_q)):
            # 保存图片
            frame = self.frame_q.pop(0)
            img_name = str(self.frame_num) + '.jpg'
            save_path = os.path.join(self.path, img_name)
            cv2.imwrite(save_path, frame)
            self.frame_num += 1
