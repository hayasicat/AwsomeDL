# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 11:43
# @Author  : ljq
# @desc    : 
# @File    : KittiLabelFormat.py
import os
import math


class KittiLabelGenerator():
    def __init__(self, data_root, label_root, dataset_name='bowling', available_subfold=[]):
        self.data_root = data_root
        self.available_subfold = available_subfold
        self.label_root = os.path.join(label_root, dataset_name)
        if not os.path.exists(self.label_root):
            os.makedirs(self.label_root)

    def process(self):
        train_files = r'train_files.txt'
        val_files = r'val_files.txt'
        date_names = os.listdir(self.data_root)
        train_list = []
        test_list = []
        for date_name in date_names:
            # 去掉去掉一些backup的文件夹名字
            if "backup" in date_name:
                continue
            if len(self.available_subfold) > 0 and (not date_name in self.available_subfold):
                continue

            date_root = os.path.join(self.data_root, date_name)
            fold_names = os.listdir(date_root)
            for fold_name in fold_names:
                subprefix = os.path.join(date_name, fold_name)
                fold_root = os.path.join(date_root, fold_name)
                if not os.path.isdir(fold_root):
                    continue
                img_names = [i.split('.')[0] for i in os.listdir(fold_root) if i.split('.')[-1] == 'png']
                if len(img_names) == 0:
                    # 对于文件夹里面没有图片的直接给pass掉
                    break
                img_names = sorted(img_names, key=lambda x: eval(x))
                # 解决排序的问题
                test_idx = math.ceil(float(len(img_names)) / float(10))
                if test_idx == 1:
                    test_idx += 1
                # 添加到列表里面,最开始的一个样本和最后的一个样本不加入列表中
                ctrain_list = [subprefix + ' ' + i for i in img_names[1:-test_idx]]
                ctest_list = [subprefix + ' ' + i for i in img_names[-test_idx:-1]]
                train_list.extend(ctrain_list)
                test_list.extend(ctest_list)
        # 写入txt文件中，考虑是否要shuffle
        with open(os.path.join(self.label_root, train_files), 'w', encoding='utf-8') as f:
            for i in train_list:
                f.write(i)
                f.write("\n")
                f.flush()

        with open(os.path.join(self.label_root, val_files), 'w', encoding='utf-8') as f:
            for i in test_list:
                f.write(i)
                f.write("\n")
                f.flush()
