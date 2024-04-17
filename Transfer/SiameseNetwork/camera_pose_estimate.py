# -*- coding: utf-8 -*-
# @Time    : 2024/4/15 15:24
# @Author  : ljq
# @desc    : 通过对比的余弦损失来判断图片是向右走还是向左走
# TODO： 1. 因为有R的存在导致说t的逆变换实际上是和R相关的。
# TODO： 2. 但是即使t只有偏移还是不行
# @File    : pose_estimate.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from torchvision import transforms
from torch.nn.modules.loss import _Loss

from Base.BackBone.TochvisionBackbone import TorchvisionResnet18
from Transfer.MonoDepth.dataset import MonoDataset


class PoseSiameseNet(nn.Module):
    # 这边通过两个conv来获取位姿的embeding向量
    def __init__(self, input_size, embedding_size):
        super(PoseSiameseNet, self).__init__()
        self.feature_encoder = TorchvisionResnet18(1)
        # embedding_side是指最后一层的vector的大小
        self.pre_stem = nn.Sequential(*[
            nn.Conv2d(input_size, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        self.next_stem = nn.Sequential(*[
            nn.Conv2d(input_size, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        # cls_head 分类头
        self.cls_head = nn.Sequential(*[
            nn.Linear(256, embedding_size),
            nn.Tanh()
        ])

    def get_pose(self, pre_features, next_features):
        features = self.pre_stem(pre_features) + self.next_stem(next_features)
        features = torch.flatten(features, 1)
        embeding_vector = self.cls_head(features)
        return embeding_vector

    def forward(self, pre_frame, cur_frame, next_frame):
        pre_features = self.feature_encoder.feature_extract(pre_frame)[-1]
        next_features = self.feature_encoder.feature_extract(next_frame)[-1]
        cur_features = self.feature_encoder.feature_extract(cur_frame)[-1]

        pos = self.get_pose(pre_features, next_features)
        neg = self.get_pose(cur_features, pre_features)
        return pos, neg


class SeqDatase(Dataset):
    def __init__(self, seq_root, start_idx=1, transform=None):
        super(SeqDatase, self).__init__()
        # 获取序列root里面的图片
        files = os.listdir(seq_root)
        files = sorted(files, key=lambda x: eval(x.split('.')[0]))[10:-1]
        # 这边
        self.file_names = files
        self.seq_root = seq_root
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # 这边返回两张图片
        cur_name = self.file_names[idx]
        # 取出前一张
        previous_name = str(eval(cur_name.split('.')[0]) - 1) + '.png'
        cur_img = cv2.imread(os.path.join(self.seq_root, cur_name), cv2.IMREAD_COLOR)[:, 16:-16, ::-1]
        previous_img = cv2.imread(os.path.join(self.seq_root, previous_name), cv2.IMREAD_COLOR)[:, 16:-16, ::-1]
        next_img = cv2.imread(os.path.join(self.seq_root, previous_name), cv2.IMREAD_COLOR)[:, 16:-16, ::-1]

        # 保持一下32的倍率
        if self.transform is None:
            return cur_img, previous_img, next_img
        cur_img = self.transform(Image.fromarray(cur_img))
        previous_img = self.transform(Image.fromarray(previous_img))
        next_img = self.transform(Image.fromarray(next_img))
        return cur_img, previous_img, next_img


augments = torchvision.transforms.Compose([
    torchvision.transforms.Resize((416, 896)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CosineLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.backend = nn.CosineSimilarity()

    def forward(self, pos, neg):
        return self.backend(pos, neg)


myloss = CosineLoss()

if __name__ == "__main__":
    device = torch.device("cuda")
    mono_data = SeqDatase(r'/root/project/AwsomeDL/data/BowlingMono/NVR_ch8_20230729070938_20230729192059/image_68/',
                          10, augments)
    train_loader = DataLoader(mono_data, batch_size=4, shuffle=True, num_workers=16)

    epoch = 100
    model = PoseSiameseNet(512, 1)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    for e in range(epoch):
        print(e)
        model.train()
        for cur_imgs, previous_imgs, next_imgs in train_loader:
            cur_imgs = cur_imgs.to(device)
            previous_imgs = previous_imgs.to(device)
            next_imgs = next_imgs.to(device)
            opt.zero_grad()
            positivate_move, negtivate_move = model(cur_imgs, previous_imgs, next_imgs)
            print(positivate_move, negtivate_move)
            cos_loss = myloss(positivate_move, negtivate_move).mean()
            print(cos_loss)
            cos_loss.backward()
            opt.step()
