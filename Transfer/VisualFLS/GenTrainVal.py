# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 14:50
# @Author  : ljq
# @desc    : root 目录下有imgs,newly,seg,backend三个文件加，newly文件夹的文件生成训练和测试集合，tran_list.txt,val_list.txt ，backend是用来存历史的标签的
# @File    : GenTrainVal.py
import os
import time
import datetime
import shutil

from Transfer.VisualFLS.CreateSegLabel import create_seg_mask
from Transfer.VisualFLS.ReWriteCorner import rewrite_corner_label


def gen_train_val_list(root_path, is_init=False):
    if is_init:
        base_fold = 'imgs'
    else:
        base_fold = 'newly'
    # 如果是初始化的话直接
    file_list = os.listdir(os.path.join(root_path, base_fold))
    img_files = [f for f in file_list if f.endswith('.jpg')]
    train_list = [f for i, f in enumerate(img_files) if i % 10 != 0]
    val_list = [f for i, f in enumerate(img_files) if i % 10 == 0]
    # 如果存在历史标签的就移动到backend目录下面
    sample_list_file = ['train.txt', 'val.txt']
    sample_list = [train_list, val_list]
    for sample_file, sample in zip(sample_list_file, sample_list):
        f_name = os.path.join(root_path, sample_file)
        if os.path.exists(f_name):
            suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S").replace('-', '')
            if not os.path.exists(os.path.join(root_path, 'backend')):
                os.makedirs(os.path.join(root_path, 'backend'))
            shutil.copyfile(f_name, os.path.join(root_path, 'backend', sample_file + suffix))
        # 创建文件
        if is_init:
            f = open(f_name, 'w', encoding='utf-8')
        else:
            f = open(f_name, 'a+', encoding='utf-8')
        f.writelines([f + '\n' for f in sample])
        f.flush()
        f.close()
    # 生成相关的seg到指定的目录
    create_seg_mask(root_path, base_fold)
    # 生成只有角点的标签到特殊的目录里面
    rewrite_corner_label(os.path.join(root_path, base_fold), os.path.join(root_path, 'seg'))


if __name__ == '__main__':
    root_path = '/root/data/VisualFLS'
    gen_train_val_list(root_path, True)
