# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 9:06
# @Author  : ljq
# @desc    : 
# @File    : dataset.py
import os
import json
import copy

import cv2
import albumentations as albu
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tt
from ThirdPack.CuUnet.HumanPts import pts2heatmap


class FLSDataset(Dataset):
    def __init__(self, root_path, is_train=True, crop_size=[420, 0, 1080, 1080], is_crop=True,
                 transforms=None, norm_transforms=None):
        # 筛选所有的图片
        self.crop_size = crop_size
        if is_crop:
            self.img_fold = os.path.join(root_path, 'imgs')
        else:
            self.img_fold = os.path.join(root_path, 'crop_imgs')
            self.crop_size = [0, 0, 10000, 10000]

        self.seg_fold = os.path.join(root_path, 'seg')
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
        return img, seg.astype(np.longlong)


# FLS_train_transforms = albu.Compose([
#     albu.Resize(640, 640),
#     albu.Cutout(num_holes=1, max_h_size=100, max_w_size=100, p=0.3),
#     # 如果有mixup
#     albu.HorizontalFlip(p=0.5),
# ])

abundle_transform = [
    albu.Resize(640, 640),
    albu.Cutout(num_holes=1, max_h_size=100, max_w_size=100, p=0.3),
    albu.HorizontalFlip(p=0.5),

    albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

    albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    albu.RandomCrop(height=512, width=512, always_apply=True),

    albu.IAAAdditiveGaussianNoise(p=0.2),
    albu.IAAPerspective(p=0.5),

    albu.OneOf(
        [
            albu.CLAHE(p=1),
            albu.RandomBrightness(p=1),
            albu.RandomGamma(p=1),
        ],
        p=0.5,
    ),

    albu.OneOf(
        [
            albu.IAASharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.5,
    ),

    albu.OneOf(
        [
            albu.RandomContrast(p=1),
            albu.HueSaturationValue(p=1),
        ],
        p=0.5,
    ),
]
FLS_train_transforms = albu.Compose(abundle_transform)
FLS_train_transforms_kp = albu.Compose(abundle_transform,
                                       keypoint_params=albu.KeypointParams(format='xy', remove_invisible=False))

FLS_test_transforms = albu.Compose([
    albu.Resize(640, 640)
])

FLS_norm_transform = albu.Normalize(mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))


def checkout_files(file_list1, file_list2):
    f1_suffix = file_list1[0].split('.')[-1]
    f2_suffix = file_list2[0].split('.')[-1]
    file_list1_head = set([f.rsplit('.')[0] for f in file_list1])
    file_list2_head = set([f.rsplit('.')[0] for f in file_list2])
    common_head = list(file_list1_head.intersection(file_list2_head))
    return [f + '.' + f1_suffix for f in common_head], [f + '.' + f2_suffix for f in common_head]


class FLSHybridDataset(FLSDataset):
    kp_types = ['ContainerSurfaceCorner', 'HoistedContainerCorner']

    # 这个是关键点检测的标签生成
    def __init__(self, root_path, is_train=True, crop_size=[420, 0, 1080, 1080], is_crop=True,
                 transforms=None, norm_transforms=None):
        super(FLSHybridDataset, self).__init__(root_path, is_train, crop_size, is_crop)
        # 筛选所有的图片
        self.transforms = transforms
        self.norm_transform = norm_transforms
        # 为了避免IO阻塞，还是先把json里面的图片给直接读出来了
        self.kps_caches = {}
        js_files = [f for f in os.listdir(self.seg_fold) if f.endswith('.json')]
        # 对比一下js_file和img_files是不是在里面，可以有img_files，但是不能只有js_files没有img_files
        js_files, img_files = checkout_files(js_files, self.img_files)
        self.img_files = img_files
        left_top = np.array(self.crop_size[:2]).reshape((-1, 2))
        for js_name in js_files:
            # 预读取
            current_kp = json.loads(open(os.path.join(self.seg_fold, js_name), 'r', encoding='utf-8').read())
            # 重新改名字一下
            new_kp = {}
            for corner_name, corner_pts in current_kp.items():
                idx = self.kp_types.index(corner_name)
                # 这里应该转为list得
                corner_pts = np.array(corner_pts).reshape((-1, 2)) - left_top
                new_kp[idx] = list(corner_pts)[0]
            self.kps_caches[js_name.split('.')[0]] = new_kp

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.img_fold, self.img_files[item]), cv2.IMREAD_COLOR)
        seg = cv2.imread(os.path.join(self.seg_fold, self.img_files[item].replace('jpg', 'png')), -1)
        x, y, w, h = self.crop_size
        img = img[y:y + h, x:x + w]
        # 读取pts
        kp_dict = copy.deepcopy(self.kps_caches[self.img_files[item].split('.')[0]])
        pt_list = []
        pt_type = []
        for k_type, pt_value in kp_dict.items():
            pt_list.append(pt_value)
            pt_type.append(int(k_type))
        if not self.transforms is None:
            # 缩放图片,这边要变换一下坐标点
            transformed = self.transforms(image=img, mask=seg, keypoints=pt_list)
            img = transformed['image']
            seg = transformed['mask']
            new_pt_list = transformed['keypoints']
        else:
            new_pt_list = pt_list
        # 开始制作heatmap图片
        heat_maps = []
        # 查找

        for i in range(2):
            if i not in pt_type:
                heat_maps.append(np.zeros((1, img.shape[0], img.shape[1]), np.float32))
                continue
            # 找到位置
            loc = pt_type.index(i)
            # print(len(pt_type), len(new_pt_list), pt_type, new_pt_list, pt_list, loc)
            pt = new_pt_list[loc]
            heat_map, _ = pts2heatmap(np.array([pt]).reshape((-1, 2)), img.shape, sigma=5)
            heat_maps.append(heat_map)
        # 堆叠在一起
        kp_heatmap = np.vstack(heat_maps)

        if not self.norm_transform is None:
            img = self.norm_transform(image=img)['image']
        img = img.transpose(2, 0, 1)

        return img, seg.astype(np.longlong), kp_heatmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #     dataset = FLSDataset('/backup/VisualFLS')
    #     for i in range(3):
    #         img,seg = dataset[i]
    #         plt.imshow(img)
    #         plt.show()
    #         seg = seg*255
    #         plt.imshow(seg)
    #         plt.show()
    import torch
    from Base.Metrics.KP import KPDis

    metric = KPDis()
    dataset = FLSHybridDataset('/backup/VisualFLS', is_crop=False, transforms=FLS_train_transforms_kp)
    for i in range(1):
        img, seg, heatmap = dataset[i]
        # 可视化看看

        metric.heatmap_dis(torch.from_numpy(heatmap), torch.from_numpy(heatmap))
        img = img.transpose(1, 2, 0)
        heatmap = heatmap.transpose(1, 2, 0)

        # 叠加看看
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(heatmap[:, :, 0])
        # plt.show()
        # plt.imshow(heatmap[:, :, 1])
        # plt.show()
        # mask = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
        # mask[:, :, 0:2] = heatmap.astype(np.uint8)
        # dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        # plt.imshow(dst)
        # plt.show()
