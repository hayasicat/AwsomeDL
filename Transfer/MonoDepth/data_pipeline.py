# -*- coding: utf-8 -*-
# @Time    : 2024/1/26 15:15
# @Author  : ljq
# @desc    : 
# @File    : test.py
from DataProcess.save_moving_fragment import handle_all_video, kitti_label_gen

video_root = r'/root/data/BowlingMono'
js_root = r'/root/data/BowlingMono/finish/'
fragment_root = r'/root/data/BowlingMono/fragments'
need_cut_info = {
    'NVR_ch8_20230729070938_20230729192059': [80, 10000],
    'NVR_ch8_20230726073037_20230726220139': [0, 140],
    'nvr238_ch8_20230801000000_20230801115457': [150, 10000],
    'nvr238_ch8_20230731000017_20230731235628': [40, 240],
    'newnvr238_ch8_20230803000011_20230803105251': [60, 75],
    'newnvr238_ch8_20230802000140_20230802115638': [80, 10000]
}

# handle_all_video(video_root, js_root, fragment_root, need_cut_info)
label_save_root = r'/root/data/BowlingMono/splits'
kitti_label_gen(fragment_root, label_save_root, dataset_name='newnvr238_ch8_20230803000011_20230803105251', available_subfold=['newnvr238_ch8_20230803000011_20230803105251'])
