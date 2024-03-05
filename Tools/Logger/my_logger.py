# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 16:08
# @Author  : ljq
# @desc    : 
# @File    : my_logger.py
import os
import logging


def init_logger(logger_path, mode='train'):
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)
    # 添加path
    handlers = []
    fold_root, file_name = os.path.split(logger_path)
    if not os.path.exists(fold_root):
        os.makedirs(fold_root)
    handlers.append(logging.FileHandler(logger_path))
    handlers.append(logging.StreamHandler())
    for hd in handlers:
        hd.setLevel(logging.INFO)
        hd.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(hd)
