# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 11:36
# @Author  : ljq
# @desc    : 
# @File    : JsonFileParser.py
import os
import json


class JosnParser():
    def __init__(self, file_path):
        self.file_path = file_path
        self.activate_interval = []
        self.subvideo_tags = {}

    def write(self, is_write_subvideo_tag=False):
        f = open(self.file_path, 'w', encoding='utf-8')
        save_dict = {"activate_interval": self.activate_interval}
        # 如果subvideo_tag的存在的话，那么也一起保存到相关的文件加里面
        if len(list(self.subvideo_tags.keys())) >= 0 and is_write_subvideo_tag:
            save_dict['subvideo_tags'] = self.subvideo_tags
        save_str = json.dumps(save_dict)
        f.write(save_str)
        f.flush()
        f.close()

    def read(self):
        f = open(self.file_path, 'r', encoding='utf-8')
        save_dict = json.loads(f.read())
        f.close()
        # 如果含有subvideo_tags的话那么就一起读取，更新一下保存的临时信息
        self.activate_interval = save_dict["activate_interval"]
        if 'subvideo_tags' in list(save_dict.keys()):
            self.subvideo_tags = save_dict['subvideo_tags']
        return save_dict["activate_interval"]

    def get_subvideo_tags(self):
        return self.subvideo_tags

    def get_activate_interval(self):
        return self.activate_interval

    def get_file_name(self):
        file_root, file_name = os.path.split(self.file_path)
        return file_name

    def fresh(self, activate_interval, subvideo_tags=dict()):
        # 默认给空值，不直接控制写入，控制类的属性来控制要写入的数据
        self.activate_interval = activate_interval
        self.subvideo_tags = subvideo_tags

    @staticmethod
    def merge_fragment(activate_interval, add_before_start=0):
        # 把多个片段给合并在一起
        new_activate_intervals = []
        for info in activate_interval:
            if len(new_activate_intervals) == 0:
                new_activate_intervals.append(info)
            # 用来合并
            cstart_frame = max(info['start_frame'] - add_before_start, 0)
            cend_frame = info['end_frame']
            # 如果发现当前片段的开始在上一片段的结束，那么合并片段
            last_end = new_activate_intervals[-1]['end_frame']
            if last_end > cstart_frame:
                new_activate_intervals[-1]['end_frame'] = cend_frame
            else:
                new_activate_intervals.append(info)
        return new_activate_intervals
