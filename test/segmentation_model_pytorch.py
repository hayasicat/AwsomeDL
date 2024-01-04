# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 12:03
# @Author  : ljq
# @desc    : 
# @File    : segmentation_model_pytorch.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

import cv2
import numpy as np
import torch
import torch.nn as nn
from VisualTask.Seg.trainner import SegTrainner
from torch.utils.data import DataLoader
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18
from Transfer.VisualFLS.dataset import FLSDataset, FLS_norm_transform, FLS_test_transforms, FLS_train_transforms

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import albumentations as albu

device = torch.device('cuda')

train_dataset = FLSDataset('/backup/VisualFLS', is_crop=False, transforms=FLS_train_transforms,
                           norm_transforms=FLS_norm_transform)
val_dataset = FLSDataset('/backup/VisualFLS', is_crop=False, is_train=False, transforms=FLS_test_transforms,
                         norm_transforms=FLS_norm_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
test_loader = DataLoader(val_dataset, batch_size=10, shuffle=True,
                         num_workers=4)

model = smp.Unet(
    encoder_name='resnet18',
    encoder_weights=None,
    classes=3)
loss = smp.utils.losses.CrossEntropyLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5, activation='argmax2d'),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])
# train_epoch = smp.utils.train.TrainEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     optimizer=optimizer,
#     device=device,
#     verbose=True,
# )
#
# valid_epoch = smp.utils.train.ValidEpoch(
#     model,
#     loss=loss,
#     metrics=metrics,
#     device=device,
#     verbose=True,
# )
# max_score = 0
# #
# for i in range(0, 40):
#
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(test_loader)
#     print(valid_logs['iou_score'])
#     # do something (save model, change lr, etc.)
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model, './best_model_None.pth')
#         print('Model saved!')
#
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')

#
best_model = torch.load('./best_model.pth')
#
pre_transform = albu.Compose([
    albu.Resize(640, 640),
    albu.Normalize(mean=(0.5, 0.5, 0.5),
                   std=(0.5, 0.5, 0.5))
])


def visual_mask(img, crop_size=[587, 402, 384, 384], is_crop=False):
    if not is_crop:
        x, y, w, h = crop_size
        img = img[y:y + h, x:x + w]
    img_tensor = pre_transform(image=img)['image']
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    pred_mask = best_model(img_tensor)
    preds = torch.softmax(pred_mask, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = preds.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
        (preds.shape[1], preds.shape[2]))
    # 得到预测图片,给原始图片着色
    img = cv2.resize(img, (640, 640))
    red = np.full(img.shape, (0, 0, 255), dtype=np.uint8)
    green = np.full(img.shape, (0, 255, 0), dtype=np.uint8)
    mask = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
    # preds = cv2.cvtColor(preds, cv2.COLOR_GRAY2BGR)
    # green_mask = cv2.bitwise_and(green, preds)
    mask[:, :, 2] = red[:, :, 2] * (preds == 1)
    mask[:, :, 1] = green[:, :, 1] * (preds == 2)
    dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    return dst
#
#
img_root_path = '/backup/VisualFLS/crop_imgs'
visual_mask_path = '/backup/VisualFLS/view_mask'
if not os.path.exists(visual_mask_path):
    os.makedirs(visual_mask_path)

container_root = r'/backup/VisualFLS/container_type'
img_files = os.listdir(container_root)
# img_files = open('/backup/VisualFLS/val.txt', 'r', encoding='utf-8').read().strip().split('\n')
for img_name in img_files:
    img = cv2.imread(os.path.join(container_root, img_name), cv2.IMREAD_COLOR)
    dst = visual_mask(img, crop_size=[587, 402, 384, 384], is_crop=False)
    # dst = SegInfere.visual(img,  is_crop=False)
    cv2.imwrite(os.path.join(visual_mask_path, img_name), dst)
