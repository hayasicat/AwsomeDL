# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 11:52
# @Author  : ljq
# @desc    : 用来保存移动时候的图片
# @File    : save_moving_fragment.py
import os, sys

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from .LabelGen.FragmentSaver import MonoDepthVideoFramgment
from .Viewer.CVIMSHOWER import ImshowWindows
from .FileHandler.JsonFileParser import JosnParser
from .FileHandler.VideoObject import CV2VideoObject
from .LabelGen.KittiLabelFormat import KittiLabelGenerator

from .FileHandler.DetectMethod import MoveDetector, SCDepthDetector


# motion_js_path = r'/backup/BowlingVideo/238NVR/C1bev/activate_interval/NVR_ch8_20230729003424_20230729052354.json'
# video_path = r'/backup/BowlingVideo/238NVR/C1bev/NVR_ch8_20230729003424_20230729052354.mp4'

def handle_all_video(video_root, js_root, fragment_root, need_cut_dict={}):
    js_files = [f.replace('.json', '') for f in os.listdir(js_root)]
    video_files = [f.replace('.mp4', '') for f in os.listdir(video_root)]

    # 找到共同的文件头
    common_files = list(set(js_files).intersection(set(video_files)))
    for file_head in common_files:
        video_path = os.path.join(video_root, file_head + '.mp4')
        js_path = os.path.join(js_root, file_head + '.json')
        # 用来保存视频片段
        save_path = os.path.join(video_root, fragment_root, file_head)
        seq_seclect = need_cut_dict.get(file_head, [])
        if len(seq_seclect) > 0:
            cut_and_save(video_path, js_path, save_path, *seq_seclect)


def kitti_label_gen(fragment_path, label_root, dataset_name='bowling', available_subfold=[]):
    if not os.path.exists(label_root):
        os.makedirs(label_root)
    KittiLabelGenerator(fragment_path, label_root, dataset_name=dataset_name,
                        available_subfold=available_subfold).process()


def cut_and_save(video_path, js_path, save_path, start_idx=0, end_idx=100000, *args):
    """
    限制保存的数量还是
    :param video_path:
    :param js_path:
    :param save_path:
    :param start_idx:
    :param end_idx:
    :return:
    """
    detector = SCDepthDetector(0.5)

    fragment_saver = MonoDepthVideoFramgment(save_path)
    activate_interval = JosnParser(js_path).read()
    video_stream = CV2VideoObject(video_path)

    # 解析保存
    for idx, info in enumerate(activate_interval):
        if start_idx <= idx <= end_idx:
            fragment_saver.new_fragment(idx)
            start_frame = max(info.get('start_frame', 0), 0)
            end_frame = info['end_frame'] - 10
            during_frame = end_frame - start_frame
            if during_frame < 0:
                continue
            # 跳转
            video_stream.locate2frame(start_frame)
            for i in range(int(during_frame)):
                frame = video_stream.get_frame()
                is_save = detector.is_activate(frame)
                if is_save:
                    fragment_saver.write(frame)
                # if i % 3 == 0:
                #     fragment_saver.write(frame)
        detector.clear()
