# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 15:33
# @Author  : ljq
# @desc    :
# @File    : fold_merge.py

import os
import time
import shutil
from shutil import copyfile

days_root = r'/backup/VisualFLS/data'
folds_root = os.listdir(days_root)
for fold_root_name in folds_root:
    fold_root = os.path.join(days_root, fold_root_name)
    date_fold_names = os.listdir(fold_root)
    for date_name in date_fold_names:
        # 把子目录的文件都给添加到一块
        date_fold_path = os.path.join(fold_root, date_name)
        if not os.path.isdir(date_fold_path):
            continue
        sub_channel_folds = os.listdir(date_fold_path)
        for sub_name in sub_channel_folds:
            sub_root = os.path.join(date_fold_path, sub_name)
            if not os.path.isdir(sub_root):
                continue

            files_names = os.listdir(sub_root)
            for files_name in files_names:
                if files_name.endswith('.txt'):
                    os.remove(os.path.join(sub_root, files_name))
                    continue
                copyfile(os.path.join(sub_root, files_name),
                         os.path.join(fold_root,
                                      fold_root_name + '_' + date_name + '_' + sub_name + '_' + files_name))
                os.remove(os.path.join(sub_root, files_name))
            shutil.rmtree(sub_root)
        shutil.rmtree(date_fold_path)

# 把主目录文件夹里面的图片按通道区分，并标记好区域，以供筛选