# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 16:58
# @Author  : ljq
# @desc    : 用来标注视频段是不是移动,临时的配置是保存在同一个目录下面的
# @File    : AnnotateActiveVideoFragment.py
import os, sys

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import cv2

from Utils.IMSHOW import ImshowWindows
from Utils.FileBackend import JosnParser
from Tools.VideoActiveDetect import VideoActivateDetector
from Tools.FragmentSplits import MonoDepthVideoFramgment

shower = ImshowWindows()


def annotate_video(video_stream, filehandler):
    # 获取一下相关的信息
    activate_interval = filehandler.get_activate_interval()
    subvideo_tags = filehandler.get_subvideo_tags()
    # print(subvideo_tags)
    # 对视频打tag
    current_subvideo_id = 0
    while current_subvideo_id < len(activate_interval):
        info = activate_interval[current_subvideo_id]
        if not str(current_subvideo_id) in list(subvideo_tags.keys()):
            subvideo_tags[current_subvideo_id] = True
        start_frame = max(info.get('start_frame', 0) - 50, 0)
        end_frame = info['end_frame']

        during_frame = end_frame - start_frame
        # 跳转
        video_stream.locate2frame(start_frame)
        add_id = True
        for i in range(int(during_frame)):
            frame = video_stream.get_frame()
            if not frame is None:
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                if current_subvideo_id != 0:
                    cv2.putText(frame, "{}:".format(current_subvideo_id - 1) + str(
                        subvideo_tags[str(current_subvideo_id - 1)]),
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                cv2.putText(frame, "{}:".format(current_subvideo_id) + str(subvideo_tags[str(current_subvideo_id)]),
                            (650, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                shower.imshow(filehandler.get_file_name(), frame)
                if i == during_frame - 1:
                    wait_time = 2000
                else:
                    wait_time = 1
                is_save = shower.fresh(wait_time)
                # 键盘事件监听
                if is_save == 115:
                    # 输入的是s则保留
                    subvideo_tags[str(current_subvideo_id)] = True
                if is_save == 100:
                    subvideo_tags[str(current_subvideo_id)] = False
                # 回溯
                if is_save == 98:
                    current_subvideo_id -= 1
                    current_subvideo_id = max(0, current_subvideo_id)
                    add_id = False
                    break
                if is_save == 110:
                    current_subvideo_id += 1
                    add_id = False
                    break
                if is_save == 32:
                    cv2.putText(frame, "{}:".format("plz press any key"),
                                (150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                    shower.fresh(0)
                if is_save == 106:
                    current_subvideo_id = len(activate_interval) + 100
                    break
        if add_id:
            current_subvideo_id += 1
        if current_subvideo_id % 10 == 0:
            # 保存一次
            filehandler.fresh(activate_interval, subvideo_tags)
            filehandler.write(True)
    # 完成写入数据，直接更新临时文件
    filehandler.fresh(activate_interval, subvideo_tags)
    filehandler.write(True)
    return filehandler


if __name__ == "__main__":
    resize_img = True

    # 视频根目录
    video_root = r'/backup/BowlingVideo/238NVR/C1bev'
    # json临时存储得位置
    temp_json_root = r'/backup/BowlingVideo/238NVR/C1bev/fragments/temp'
    # 完全做完存储的json的位置
    finish_json_root = r'/backup/BowlingVideo/238NVR/C1bev/fragments/finish'
    # 视频片段保存的路径
    video_save_root = r'/backup/BowlingVideo/238NVR/C1bev/fragments'

    if not os.path.exists(temp_json_root):
        os.makedirs(temp_json_root)
    if not os.path.exists(finish_json_root):
        os.makedirs(finish_json_root)

    video_names = os.listdir(video_root)
    detector = VideoActivateDetector()
    for video_name in video_names:
        if video_name.rsplit('.', 1)[-1] != 'mp4':
            continue
        # 用来切分视频段。如果有的
        video_js_name = video_name.rsplit('.', 1)[0] + '.json'
        temp_json_path = os.path.join(temp_json_root, video_js_name)
        finish_json_path = os.path.join(finish_json_root, video_js_name)
        is_detected = False
        if os.path.isfile(temp_json_path):
            is_detected = True
        filehandler = JosnParser(temp_json_path)
        video_path = os.path.join(video_root, video_name)
        video_stream = detector.open(video_path)
        if not is_detected:
            # 这个检测的过程实在是太耗时了
            activate_interval = detector.detect(video_stream)
            # 获取活动的区间，用配置文件解析的方式来给每段视频打标注
            subvideo_tags = {i: True for i in range(len(activate_interval))}
            filehandler.fresh(activate_interval, subvideo_tags)
            filehandler.write(True)
        else:
            # 因为没有read导致做完跑的json失效了
            filehandler.read()
        print(temp_json_path, "写入完成")
        #
        # 构造活跃的标签
        # filehandler = annotate_video(video_stream, filehandler)
        activate_interval = filehandler.get_activate_interval()
        subvideo_tags = filehandler.get_subvideo_tags()
        # # 写入活跃的标签，并且
        new_activate_interval = []
        for idx, info in enumerate(activate_interval):
            if subvideo_tags[str(idx)]:
                new_activate_interval.append(info)

        # 设定好图片的保存地址
        video_save_path = os.path.join(video_save_root, video_name.rsplit('.', 1)[0])
        saver = MonoDepthVideoFramgment(video_save_path)
        for info in new_activate_interval:
            start_frame = max(info.get("start_frame", 0) - 10, 0)
            end_frame = info['end_frame']
            during_frame = end_frame - start_frame
            # 跳转
            video_stream.locate2frame(start_frame)
            for i in range(int(during_frame)):
                frame = video_stream.get_frame()
                # shower.imshow("motion_fragment", frame)
                # is_save = shower.fresh()
                if i % 7 == 0:
                    if resize_img == True:
                        frame = frame[:, 16:-16, :]
                        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                    saver.write(frame)
            saver.new_fragment()
        # 切分数据集
        finish_handler = JosnParser(finish_json_path)
        finish_handler.fresh(new_activate_interval)
        finish_handler.write()
        # 切分完请按任意键进入下一个视频
        # cv2.waitKey(1)
