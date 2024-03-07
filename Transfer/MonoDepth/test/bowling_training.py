# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 17:10
# @Author  : ljq
# @desc    : 
# @File    : bowling_training.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from Transfer.MonoDepth.MonoTrainer.TripleImages import TripleTrainer
from Transfer.MonoDepth.MonoTrainer.PairImages import PairTrainer

from Transfer.MonoDepth.dataset import MonoDataset
from Base.BackBone.STN import MonoDepthSTN, MonoDepthPair
from Base.BackBone.EfficientNetV2 import EfficientNetV2S
from Base.BackBone.ResNet import ResNet18, ResNet34
from Base.BackBone.TochvisionBackbone import TorchvisionResnet18
from Base.SegHead.DepthHead import DepthDecoder, DepthNet
from Base.SegHead.MonoDepth import DepthHead, MonoDepth
from Base.SegHead.PAN import PANDecoder

data_root = r'/root/project/AwsomeDL/data/BowlingMono'
# data_root = r'/root/project/AwsomeDL/data/BowlingMonoNew'

train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
# train_file_path = os.path.join(data_root, r'splits/bowling/train_files.txt')

# data_root = r'/root/data/BowlingMono'
# train_file_path = os.path.join(data_root, r'splits/newnvr238_ch8_20230803000011_20230803105251/train_files.txt')
#
# img_fold = os.path.join(data_root, 'fragments')
# data_root = img_fold

# train_data = MonoDataset(data_root, train_file_path, 416, 896)
# train_data = MonoDataset(data_root, train_file_path, 832, 1824)
train_data = MonoDataset(data_root, train_file_path, 416, 896, coor_shift=[16, 0])

# 相比于姿态估计网络，backbone换轻量级的倒是比较无所谓
# encoder = EfficientNetV2S(10, input_chans=3)
# encoder = ResNet34(10, input_chans=3)

# encoder = ResNet18(10, input_chans=3)
# 使用预训练的encoder
encoder = TorchvisionResnet18(2, input_chans=3)
depth_decoder = PANDecoder(encoder.channels[::-1])
depth_net = MonoDepth(encoder, depth_decoder)

# depth_decoder = DepthDecoder(encoder.channels)
# depth_net = DepthNet(encoder, depth_decoder)
# TODO: monodepth2预测出来的是两张图片两张深度，但是我这边就直接两张图片一个姿态

# 不能那比较难收敛的轻量级网络来训姿态估计网络
# model = MonoDepthSTN(depth_net, ResNet18)
model = MonoDepthSTN(depth_net, ResNet18)

trainer = TripleTrainer(train_data, model, is_parallel=False)
# trainer.train()

# sc_depth方式训练
# model = MonoDepthPair(depth_net, ResNet18)
# trainer = PairTrainer(train_data, model)
# trainer.train()

# 分析一下结果
# model_path = r'/root/project/AwsomeDL/data/monodepth/total_fragment.pth'
# model_path = r'/root/project/AwsomeDL/data/monodepth/temp_90_model.pth'

# model_path = '/root/project/AwsomeDL/data/baseline/90_model.pth'
# model_path = r'/root/project/AwsomeDL/data/monodepth/geometry_consistance.pth'


# pair-model
# model_path = r'/root/project/AwsomeDL/data/baseline/90_model.pth'
# trainer.resume_from(model_path)
# trainer.analys()
# trainer.recorder()
