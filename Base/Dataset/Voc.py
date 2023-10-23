# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 11:15
# @Author  : ljq
# @desc    : 
# @File    : Voc.py
import torchvision
from PIL import Image

class MyVocDataset(torchvision.datasets.VOCSegmentation):
    def __init__(self,
                 root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 ):
        super().__init__(root, year, image_set, download, transform, target_transform, transform)

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
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        # torchvision的transformer对语义分割不太友好
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
