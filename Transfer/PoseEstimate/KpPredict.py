# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 14:15
# @Author  : ljq
# @desc    :  关键点预测
# @File    : KpPredict.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import torch
import numpy as np
import albumentations as albu

import os
import cv2
import matplotlib.pyplot as plt
from Base.SegHead.Unet import Unet, UnetHead
from Base.BackBone import ResNet34, ResNet18

pre_transform = albu.Compose([
    albu.Resize(640, 640),
    albu.Normalize(mean=(0.5, 0.5, 0.5),
                   std=(0.5, 0.5, 0.5))
])


class MultiHeadInference():
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device

    def visual(self, img):
        img_tensor = pre_transform(image=img)['image']
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask, heatmap = self.model(img_tensor)
        heatmap = heatmap.squeeze()
        heatmap = heatmap.detach().cpu().permute(1, 2, 0).numpy()
        print(np.max(heatmap), np.mean(heatmap))
        plt.imshow(heatmap[:, :, 0])
        plt.title('hoist' + str(np.max(heatmap[:, :, 0])))
        plt.show()
        plt.imshow(heatmap[:, :, 1])
        plt.title('container: ' + str(np.max(heatmap[:, :, 1])))
        plt.show()


if __name__ == "__main__":
    encoder = ResNet34(20, small_scale=False)
    decoder = UnetHead()
    model = Unet(encoder, decoder, 3, 2, activation='')
    model.load_state_dict(torch.load('../../data/lockhole/multi_head/100_model.pth'))
    model = model.to(torch.device("cuda:0"))
    model.eval()
    SegInfere = MultiHeadInference(model, torch.device("cuda:0"))
    img_root_path = '/backup/VisualFLS/crop_imgs'
    img_files = os.listdir(img_root_path)
    for idx, img_name in zip(range(10), img_files):
        img = cv2.imread(os.path.join(img_root_path, img_name), cv2.IMREAD_COLOR)
        # img = cv2.imread(os.path.join(img_root_path, img_name), cv2.IMREAD_COLOR)
        dst = SegInfere.visual(img)
        img = cv2.resize(img, (640, 640))
        plt.imshow(img)
        plt.show()
