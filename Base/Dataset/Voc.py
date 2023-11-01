# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 11:15
# @Author  : ljq
# @desc    : 
# @File    : Voc.py
import torchvision
import numpy as np
from PIL import Image


class MyVocDataset(torchvision.datasets.VOCSegmentation):
    def __init__(self,
                 root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform=None,
                 norm_transform=None,
                 ):
        super().__init__(root, year, image_set, download)
        self.transforms = transform
        self.norm_transform = norm_transform

    @property
    def masks(self):
        return self.targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = np.array(Image.open(self.images[index]).convert("RGB"), dtype=np.uint8)
        target = np.array(Image.open(self.masks[index]), np.uint8)
        # 短边缩放

        # torchvision的transformer对语义分割不太友好
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=target)
            img = transformed['image']
            target = transformed['mask']
        if self.norm_transform is not None:
            img = self.norm_transform(image=img)['image']
        img = img.transpose(2, 0, 1)
        target[target == 255] = 0
        return img, target
