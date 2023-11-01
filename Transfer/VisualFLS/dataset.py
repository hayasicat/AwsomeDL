# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 9:06
# @Author  : ljq
# @desc    : 
# @File    : dataset.py
import os

import cv2
import albumentations as albu
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tt


class FLSDataset(Dataset):
    def __init__(self, root_path, is_train=True, crop_size=[420, 0, 1080, 1080], transforms=None, norm_transforms=None):
        # 筛选所有的图片
        self.img_fold = os.path.join(root_path, 'imgs')
        self.seg_fold = os.path.join(root_path, 'seg')
        self.crop_size = crop_size
        self.transforms = transforms
        self.norm_transform = norm_transforms
        self.is_train = is_train
        if self.is_train:
            base_file = 'train.txt'
        else:
            base_file = 'val.txt'
        # img ,seg目录下的文件和imgs下的同名
        files = open(os.path.join(root_path, base_file), 'r', encoding='utf-8').read()
        self.img_files = files.strip().split('\n')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.img_fold, self.img_files[item]), cv2.IMREAD_COLOR)
        seg = cv2.imread(os.path.join(self.seg_fold, self.img_files[item].replace('jpg', 'png')), -1)
        x, y, w, h = self.crop_size
        img = img[y:y + h, x:x + w]
        if not self.transforms is None:
            # 缩放图片
            transformed = self.transforms(image=img, mask=seg)
            img = transformed['image']
            seg = transformed['mask']
        if not self.norm_transform is None:
            img = self.norm_transform(image=img)['image']
        img = img.transpose(2, 0, 1)
        return img, seg


FLS_train_transforms = albu.Compose([
    albu.Resize(640, 640),
    albu.Cutout(num_holes=1, max_h_size=100, max_w_size=100, p=0.3),
    # 如果有mixup
    albu.HorizontalFlip(p=0.5),
])

FLS_test_transforms = albu.Compose([
    albu.Resize(640, 640)
])

FLS_norm_transform = albu.Normalize(mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))

# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     dataset = FLSDataset('/backup/VisualFLS')
#     for i in range(3):
#         img,seg = dataset[i]
#         plt.imshow(img)
#         plt.show()
#         seg = seg*255
#         plt.imshow(seg)
#         plt.show()
