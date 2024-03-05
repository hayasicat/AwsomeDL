# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 10:08
# @Author  : ljq
# @desc    : 
# @File    : Inference.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import torch
import numpy as np
import albumentations as albu
from Base.Visual.SegView import SegViewer


class SegInference():
    def __init__(self, model, device, transform=None, norm_transform=None):
        self.model = model
        self.model.eval()
        self.device = device
        self.viewer = SegViewer()
        # 注册推理的钩子
        # TODO： 钩子的部分不应该自己手动注册，应该在模型推理的时候定义好了
        self.transform = transform
        self.norm_transform = norm_transform
        self.inf_heads = []

    def register(self, func):
        self.inf_heads.append(func)

    def kp_head(self, img, pred):
        pred = pred.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        dst = self.viewer.kp_view(img, pred)
        return dst

    def cls_head(self, img, pred):
        color_ = np.array([
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0]], np.uint8)

        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
            (pred.shape[1], pred.shape[2]))
        dst = self.viewer.cls_view(img, pred, color=color_)
        return dst

    def inference(self, img, crop_size=[420, 0, 1080, 1080], is_crop=False, is_draw=False):
        if not is_crop:
            x, y, w, h = crop_size
            img = img[y:y + h, x:x + w]
        if not self.transform is None:
            img = self.transform(image=img)['image']
        img_tensor = img
        if not self.norm_transform is None:
            img_tensor = self.norm_transform(image=img_tensor)['image']
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        preds = self.model(img_tensor)
        if not is_draw:
            return preds
        result = []
        for head_func, pred in zip(self.inf_heads, preds):
            result.append(head_func(img, pred))
        return result


if __name__ == '__main__':
    import os
    import cv2
    import matplotlib.pyplot as plt
    from Base.SegHead.Unet import Unet, UnetHead
    from Base.BackBone import ResNet34, ResNet18, EfficientNetV2S
    from Base.BackBone.TochvisionBackbone import TorchvisionResnet18
    from Transfer.VisualFLS.dataset import FLS_test_transforms, FLS_norm_transform

    # encoder = EfficientNetV2S(20)
    encoder = TorchvisionResnet18(2)
    decoder = UnetHead(encoder.channels[::-1])

    # decoder = UnetHead()
    model = Unet(encoder, decoder, 3, 2, activation='')
    # model.load_state_dict(torch.load('../../data/lockhole/multi_head/EFUnet_TC2/200_model.pth'))
    model.load_state_dict(torch.load('../../data/lockhole/multi_head/torchUnet_TC4/last.pth'))

    model = model.to(torch.device("cuda:0"))
    model.eval()
    seg_inf = SegInference(model, torch.device("cuda:0"), FLS_test_transforms, FLS_norm_transform)
    seg_inf.register(seg_inf.cls_head)
    seg_inf.register(seg_inf.kp_head)
    img_root_path = '/root/data/VisualFLS/crop_imgs'
    visual_mask_path = '/root/data/VisualFLS/view_mask'
    if not os.path.exists(visual_mask_path):
        os.makedirs(visual_mask_path)
    # img_files = [f for f in os.listdir(img_root_path) if f.endswith('.jpg')]
    # img_files = open('/backup/VisualFLS/val.txt', 'r', encoding='utf-8').read().strip().split('\n')
    # for img_name in img_files:
    #     img = cv2.imread(os.path.join(img_root_path, img_name), cv2.IMREAD_COLOR)
    #     dst = SegInfere.visual(img, is_crop=True)
    #     cv2.imwrite(os.path.join(visual_mask_path, img_name), dst)
    # plt.imshow(dst)
    # plt.show()
    # 箱型检测
    # img_root_path = r'/backup/VisualFLS/container_type'
    # img_files = os.listdir(img_root_path)
    img_files = [f for f in os.listdir(img_root_path) if f.endswith('.jpg')]
    # img_files = open('/backup/VisualFLS/val.txt', 'r', encoding='utf-8').read().strip().split('\n')
    for img_name in img_files:
        img = cv2.imread(os.path.join(img_root_path, img_name), cv2.IMREAD_COLOR)
        # seg_res, kp_res = seg_inf.inference(img, crop_size=[587, 402, 384, 384], is_crop=False, is_draw=True)
        seg_res, kp_res = seg_inf.inference(img, crop_size=[587, 402, 384, 384], is_crop=True, is_draw=True)

        # dst = SegInfere.visual(img,  is_crop=True)
        # plt.imshow(seg_res)
        # plt.show()
        cv2.imwrite(os.path.join(visual_mask_path, img_name), seg_res)
