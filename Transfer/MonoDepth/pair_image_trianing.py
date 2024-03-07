# -*- coding: utf-8 -*-
# @Time    : 2024/1/16 19:08
# @Author  : ljq
# @desc    : 
# @File    : pair_image_trianing.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from Transfer.MonoDepth.MonoTrainer.TripleImages import TripleTrainer
from Transfer.MonoDepth.MonoTrainer.PairImages import PairTrainer

from Transfer.MonoDepth.dataset import MonoDataset
from Base.BackBone.STN import MonoDepthSTN, MonoDepthPair
from Base.BackBone.EfficientNetV2 import EfficientNetV2S
from Base.BackBone.ResNet import ResNet18, ResNet34
from Base.BackBone.TochvisionBackbone import TorchvisionResnet18
from Base.SegHead.MonoDepth import DepthHead, MonoDepth
from Base.SegHead.PAN import PANDecoder

from Base.SegHead.DepthHead import DepthDecoder, DepthNet

data_root = r'/root/project/AwsomeDL/data/BowlingMono'
train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
fragment_path = data_root
#
# data_root = r'/root/data/BowlingMono'
# train_file_path = os.path.join(data_root, r'splits/newnvr238_ch8_20230803000011_2023080310525106/train_files.txt')
# fragment_path = os.path.join(data_root, 'fragments')

# train_data = MonoDataset(fragment_path, train_file_path, 384, 864, coor_shift=[48, 32])
# 可以增加一个crop的增强
train_data = MonoDataset(fragment_path, train_file_path, 416, 896, coor_shift=[16, 0])

# train_data = MonoDataset(data_root, train_file_path, 832, 1824)

# encoder = ResNet18(10, input_chans=3)
# encoder = TorchvisionResnet18(10, input_chans=3)
# depth_decoder = DepthDecoder(encoder.channels)
# depth_net = DepthNet(encoder, depth_decoder)
# TODO: monodepth2预测出来的是两张图片两张深度，但是我这边就直接两张图片一个姿态
#
encoder = TorchvisionResnet18(2, input_chans=3)
depth_decoder = PANDecoder(encoder.channels[::-1])
depth_net = MonoDepth(encoder, depth_decoder)

# sc_depth方式训练
# model = MonoDepthSTN(depth_net, ResNet18)
model = MonoDepthSTN(depth_net, ResNet18)

trainer = PairTrainer(train_data, model)
trainer.train()

# 分析一下结果
# model_path = r'/root/project/AwsomeDL/data/sc_depth/without_pretrain_model.pth'
# model_path = r'/root/project/AwsomeDL/data/monodepth/bucket_model.pth'
# model_path = r'/root/project/AwsomeDL/data/monodepth/geometry_consistance.pth'

# pair-model

# model_path = r'/root/project/AwsomeDL/data/sc_depth/without_pretrain_model.pth'
# model_path = r'/root/project/AwsomeDL/data/sc_depth/best.pth'
# trainer.resume_from(model_path)
# trainer.analys()
# trainer.recorder()
# trainer.eval()
