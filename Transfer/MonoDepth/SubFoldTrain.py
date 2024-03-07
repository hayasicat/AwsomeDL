# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 15:05
# @Author  : ljq
# @desc    : 
# @File    : SubFoldTrain.py

import os
import json
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

from Transfer.MonoDepth.MonoTrainer.TripleImages import TripleTrainer
from Transfer.MonoDepth.MonoTrainer.PairImages import PairTrainer

from Transfer.MonoDepth.dataset import MonoDataset, MonoDatasetFold
from Base.BackBone.STN import MonoDepthSTN, MonoDepthPair
from Base.BackBone.EfficientNetV2 import EfficientNetV2S
from Base.BackBone.ResNet import ResNet18, ResNet34
from Base.BackBone.TochvisionBackbone import TorchvisionResnet18
from Base.SegHead.MonoDepth import DepthHead, MonoDepth
from Base.SegHead.PAN import PANDecoder

from Base.SegHead.DepthHead import DepthDecoder, DepthNet

from Tools.Logger.my_logger import init_logger


# data_root = r'/root/project/AwsomeDL/data/BowlingMono'
# train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
# fragment_path = data_root


class ExperimentsLogger():
    def __init__(self, file_path, resume=False):
        """
        用来判断是不是需要跑，记录一下最终的结果是什么
        :param path:
        """
        self.train_result = {}
        self.file_path = file_path
        if resume:
            self.resume_data(file_path)

    def resume_data(self, file_path):
        self.train_result = json.loads(open(file_path, 'r').read())

    def fresh_data(self):
        with open(self.file_path, 'w') as f:
            f.write(json.dumps(self.train_result))
            f.flush()

    def update(self, sub_fold, loss):
        self.train_result[sub_fold] = str(loss)

    def is_recorded(self, sub_fold):
        k_names = list(self.train_result.keys())
        if sub_fold in k_names:
            return True
        else:
            return False


encoder = TorchvisionResnet18(2, input_chans=3)
depth_decoder = PANDecoder(encoder.channels[::-1])
depth_net = MonoDepth(encoder, depth_decoder)
model = MonoDepthSTN(depth_net, ResNet18)

experiments_root = '/root/project/AwsomeDL/data/sub_fold'
weight_path = os.path.join(experiments_root, 'weights')

# 保存训练日志
log_path = os.path.join(experiments_root, 'train.log')
finish_seqs = os.path.join(experiments_root, 'finish_seqs.json')
# 显示每个seq的训练结果
visual_results = os.path.join(experiments_root, 'visual')

if not os.path.exists(weight_path):
    os.makedirs(weight_path)
if not os.path.exists(visual_results):
    os.makedirs(visual_results)

data_root = r'/root/data/BowlingMono/fragments'

# 保存最初始的日志
torch.save(model.state_dict(), os.path.join(weight_path, 'init.pth'))
init_logger(log_path)
logger = logging.getLogger('train')

sub_videos = os.listdir(data_root)

exp_log = ExperimentsLogger(finish_seqs, resume=True)

for sub_video in sub_videos:
    video_root = os.path.join(data_root, sub_video)
    sub_seqs = os.listdir(video_root)
    for seq in sub_seqs:
        sub_fold = os.path.join(sub_video, seq)
        if exp_log.is_recorded(sub_fold):
            continue
        logger.info("current fold {}".format(sub_fold))
        # 创建数据集，记录日志
        train_data = MonoDatasetFold(data_root, sub_fold, 416, 896, coor_shift=[16, 0])
        # 重头开始一个训练
        img_save_root = os.path.join(visual_results, sub_fold)
        trainer = PairTrainer(train_data, model, model_path=os.path.join(weight_path, 'init.pth'), save_pth='',
                              log_path=log_path, is_init=True, use_plt=False, img_save_root=img_save_root)
        trainer.train()
        trainer.analys()
        result_loss = trainer.eval()
        # 记录相关的值
        exp_log.update(sub_fold, loss=result_loss.detach().cpu().numpy()[0])
        exp_log.fresh_data()

    # 清楚一下训练的参数
