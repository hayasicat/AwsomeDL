# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 17:00
# @Author  : ljq
# @desc    : 
# @File    : trainner.py
import os


# 如果最后一层是torch.sigmoid的话，不如给一个nn.relu6合适，然后缩放nn.relu6到0-1
class MonoTrainer():
    def __init__(self):
        pass


if __name__ == "__main__":
    from Transfer.MonoDepth.dataset import MonoDataset

    data_root = r'/root/project/AwsomeDL/data/BowlingMono'
    train_file_path = os.path.join(data_root, r'bowling/train_files.txt')
    train_data = MonoDataset(data_root, train_file_path, 832, 1824)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_data, 4)
    # 模型初始化
    from Base.BackBone.STN import MonoDepthSTN
    from Base.BackBone.EfficientNetV2 import EfficientNetV2S
    from Base.BackBone.ResNet import ResNet18
    from Base.SegHead.DepthHead import DepthDecoder, DepthNet

    # 初始化以后
    encoder = EfficientNetV2S(10, input_chans=3)
    # encoder = ResNet34(10, input_chans=3)
    depth_decoder = DepthDecoder(encoder.channels)
    depth_net = DepthNet(encoder, depth_decoder)

    model = MonoDepthSTN(depth_net, ResNet18)
    for data in train_loader:
        depth_maps, refers_trans, next_trans = model(data)
        # 转换参数
        break
