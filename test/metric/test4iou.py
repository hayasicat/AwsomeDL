# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 10:44
# @Author  : ljq
# @desc    : 
# @File    : test4iou.py
import os

import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18
from Base.Metrics.SEG import IOU
from VisualTask.Seg.Inference import SegInference
from Transfer.VisualFLS.dataset import FLSDataset, FLS_test_transforms, FLS_norm_transform

val_dataset = FLSDataset('/backup/VisualFLS', is_train=False, transforms=FLS_test_transforms,
                         norm_transforms=FLS_norm_transform)
test_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)

encoder = ResNet34(20, small_scale=False)
decoder = UnetHead(2, activation='')
model = Unet(encoder, decoder)
model.load_state_dict(torch.load('../../data/lockhole/190_unet_res34.pth'))
model = model.to(torch.device("cuda:0"))
model.eval()
for x, gts in test_loader:
    with torch.no_grad():
        x = x.to(torch.device("cuda:0"))
        gts = gts.to(torch.device("cuda:0"))
        preds = model(x)
        IOU.iou(preds, gts)

# SegInfere = SegInference(model, torch.device("cuda:0"))
# img_root_path = '/backup/VisualFLS/imgs'
# visual_mask_path = '/backup/VisualFLS/view_mask'
# img_files = [f for f in os.listdir(img_root_path) if f.endswith('.jpg')]
# img_files = open('/backup/VisualFLS/val.txt', 'r', encoding='utf-8').read().strip().split('\n')
# for img_name in img_files:
#     img = cv2.imread(os.path.join(img_root_path, img_name), cv2.IMREAD_COLOR)
#     gt = cv2.imread(os.path.join(img_root_path.replace('imgs', 'seg'), img_name.replace('jpg', 'png')), -1)
#     gt = cv2.resize(gt, (640, 640))
#     pred = SegInfere.inference(img).detach().cpu()
#     # 如何读取
#     IOU.iou(pred, gt)
#
#     dst = SegInfere.visual(img)
#     cv2.imwrite(os.path.join(visual_mask_path, img_name), dst)
#     plt.imshow(dst)
#     plt.show()

# 获取数据集
