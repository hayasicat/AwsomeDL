# -*- coding: utf-8 -*-
# @Time    : 2024/2/28 14:13
# @Author  : ljq
# @desc    : 
# @File    : LabelHandler.py
import os
import datetime
from shutil import copyfile

from Transfer.LandSideTask.DataProcess.YHProcessor import YHProcessor


class LabelHandler():
    img_suffixs = ['.jpg', '.png']
    source_img_path = 'images'
    newly_img_path = 'newly'
    save_label_path = 'labels'
    train_labels = 'train.txt'
    val_labels = 'val.txt'
    backup_path = 'backup'

    def __init__(self, dataset_root):
        cls_index = 'index.txt'
        # 获取标注的类别，按顺序构建yolo数据
        img_root = os.path.join(dataset_root, self.source_img_path)
        self.cls_list = open(os.path.join(img_root, cls_index), 'r', encoding='utf-8').read().strip().split(',')
        self.data_root = dataset_root
        self.processor = {}
        self.processor['YH'] = YHProcessor(self.cls_list)

    def handle(self):
        # 用来处理数据
        img_head = self.source_img_path
        if self.is_have_train_file(self.data_root):
            img_head = self.newly_img_path
            self.backup(os.path.join(self.data_root, self.train_labels))
            self.backup(os.path.join(self.data_root, self.val_labels))
        # 进行更新
        file_list = os.listdir(os.path.join(self.data_root, img_head))
        img_files = [f for f in file_list if f.endswith('.jpg')]
        train_list = [f for i, f in enumerate(img_files) if i % 10 != 0]
        val_list = [f for i, f in enumerate(img_files) if i % 10 == 0]
        # 补充更新标签
        self.fresh_dataset(train_list, val_list, self.data_root, img_head)
        self.generate_labels(self.data_root, img_files, img_head)

    def generate_labels(self, data_root, img_files, img_head):
        """
        用来转换标签
        :return:
        """
        save_root = os.path.join(data_root, self.save_label_path)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for img_name in img_files:
            img_path = os.path.join(data_root, img_head, img_name)
            # 写入生成的标签
            content = self.processor['YH'].transform(img_path)
            # 替换最后一个.
            label_name = img_name.rsplit('.')[0] + '.txt'
            # 生成格式
            with open(os.path.join(save_root, label_name), 'w', encoding='utf-8') as f:
                f.write(content)

    def fresh_dataset(self, train_list, val_list, data_root, img_root):
        """
        更新训练数据集
        :param train_list:
        :param val_list:
        :param img_root:
        :return:
        """
        is_init = not self.is_have_train_file(data_root)
        sample_list_file = [self.train_labels, self.val_labels]
        sample_list = [train_list, val_list]
        for sample_file, sample in zip(sample_list_file, sample_list):
            f_name = os.path.join(data_root, sample_file)
            # 创建文件
            if is_init:
                f = open(f_name, 'w', encoding='utf-8')
            else:
                f = open(f_name, 'a+', encoding='utf-8')
            f.writelines([os.path.join('.', img_root, f) + '\n' for f in sample])
            f.flush()
            f.close()

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


if __name__ == "__main__":
    LabelHandler(r'G:\FLSNEW\example').handle()
