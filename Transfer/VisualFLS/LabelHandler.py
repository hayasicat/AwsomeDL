# -*- coding: utf-8 -*-
# @Time    : 2024/1/24 17:25
# @Author  : ljq
# @desc    : 
# @File    : LabelHandler.py
import os
import time
import json
import datetime
from shutil import copyfile

import cv2

from Transfer.VisualFLS.DataProcess import YHProcessor, TCProcessor


class LabelHandler:
    seg_save_path = 'seg'
    crop_imgs_save_path = 'crop_imgs'
    source_img_path = 'imgs'
    newly_img_path = 'newly'
    train_labels = 'train.txt'
    val_labels = 'val.txt'
    backup_path = 'backup'

    def __init__(self):
        self.processor = {}
        self.processor['YH'] = YHProcessor()
        self.processor['TC'] = TCProcessor()

    def is_have_train_file(self, img_root):
        if os.path.exists(os.path.join(img_root, self.train_labels)) \
                and os.path.exists(os.path.join(img_root, self.val_labels)):
            return True
        return False

    def backup(self, file_path):
        file_root, file_name = os.path.split(file_path)
        if not os.path.exists(os.path.join(file_root, self.backup_path)):
            os.makedirs(os.path.join(file_root, self.backup_path))
        suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S").replace('-', '')
        copyfile(file_path, os.path.join(file_root, self.backup_path, file_name + suffix))

    def fresh_dataset(self, train_list, val_list, img_root):
        """
        更新训练数据集
        :param train_list:
        :param val_list:
        :param img_root:
        :return:
        """
        is_init = not self.is_have_train_file(img_root)
        sample_list_file = [self.train_labels, self.val_labels]
        sample_list = [train_list, val_list]
        for sample_file, sample in zip(sample_list_file, sample_list):
            f_name = os.path.join(img_root, sample_file)
            # 创建文件
            if is_init:
                f = open(f_name, 'w', encoding='utf-8')
            else:
                f = open(f_name, 'a+', encoding='utf-8')
            f.writelines([f + '\n' for f in sample])
            f.flush()
            f.close()

    def handle(self, img_root):
        # 如果存在train.txt和val.txt 那么就移动到backup备份
        img_head = self.source_img_path
        if self.is_have_train_file(img_root):
            img_head = self.newly_img_path
            self.backup(os.path.join(img_root, self.train_labels))
            self.backup(os.path.join(img_root, self.val_labels))
        # 进行更新
        file_list = os.listdir(os.path.join(img_root, img_head))
        img_files = [f for f in file_list if f.endswith('.jpg')]
        train_list = [f for i, f in enumerate(img_files) if i % 10 != 0]
        val_list = [f for i, f in enumerate(img_files) if i % 10 == 0]
        # 补充更新标签
        self.fresh_dataset(train_list, val_list, img_root)
        self.generate_labels(img_root, file_list, img_head)

    def generate_labels(self, img_root, file_list, img_head):
        img_files = [f for f in file_list if f.endswith('jpg')]
        seg_root = os.path.join(img_root, self.seg_save_path)
        crop_root = os.path.join(img_root, self.crop_imgs_save_path)
        js_root = os.path.join(img_root, self.seg_save_path)
        for img_name in img_files:
            if 'tc' in img_name[:3]:
                processor = self.processor['TC']
            else:
                processor = self.processor['YH']
            # 解决问题
            result = processor.transform(os.path.join(img_root, img_head, img_name))
            # 保存标签
            crop_img = result['img_patch']
            mask_img = result['mask_patch']
            corner_points_js = result['corner_points']
            cv2.imwrite(os.path.join(seg_root, img_name.replace('jpg', 'png')), mask_img)
            cv2.imwrite(os.path.join(crop_root, img_name), crop_img)
            with open(os.path.join(js_root, img_name.replace('.jpg', '.json')), 'w', encoding='utf-8') as f:
                f.write(json.dumps(corner_points_js))
                f.flush()


if __name__ == '__main__':
    LabelHandler().handle(r'/root/data/VisualFLS')
