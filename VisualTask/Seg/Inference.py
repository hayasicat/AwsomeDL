# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 10:08
# @Author  : ljq
# @desc    : 
# @File    : Inference.py
import cv2
import torch
import numpy as np
import albumentations as albu

pre_transform = albu.Compose([
    albu.Resize(640, 640),
    albu.Normalize(mean=(0.5, 0.5, 0.5),
                   std=(0.5, 0.5, 0.5))
])


class SegInference():
    def __init__(self, model, device):
        self.model = model
        self.model.eval()
        self.device = device

    def visual(self, img, crop_size=[420, 0, 1080, 1080], target=None):
        """
        缩放以后再将feature map缩放回去
        :param img:
        :param target:
        :return:
        """
        x, y, w, h = crop_size
        img = img[y:y + h, x:x + w]
        img_tensor = pre_transform(image=img)['image']
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred_mask = self.model(img_tensor)
        preds = torch.softmax(pred_mask, dim=1)
        preds = torch.argmax(preds, dim=1)
        preds = preds.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8).reshape(
            (preds.shape[1], preds.shape[2]))
        # 得到预测图片,给原始图片着色
        img = cv2.resize(img, (640, 640))
        green = np.full(img.shape, (0, 0, 255), dtype=np.uint8)
        # preds = cv2.cvtColor(preds, cv2.COLOR_GRAY2BGR)
        # green_mask = cv2.bitwise_and(green, preds)
        green[:, :, 2] = green[:, :, 2] * preds
        dst = cv2.addWeighted(img, 0.5, green, 0.5, 0)
        return dst

    def inference(self, img, crop_size=[420, 0, 1080, 1080]):
        x, y, w, h = crop_size
        img = img[y:y + h, x:x + w]
        img_tensor = pre_transform(image=img)['image']
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred_mask = self.model(img_tensor).squeeze(0)
        return pred_mask


if __name__ == '__main__':
    import os
    import cv2
    import matplotlib.pyplot as plt
    from Base.SegHead.Unet import Unet, UnetHead
    from Base.BackBone import ResNet34, ResNet18

    encoder = ResNet34(20, small_scale=False)
    decoder = UnetHead(2, activation='')
    model = Unet(encoder, decoder)
    model.load_state_dict(torch.load('../../data/lockhole/190_unet_res34.pth'))
    model = model.to(torch.device("cuda:0"))
    model.eval()
    SegInfere = SegInference(model, torch.device("cuda:0"))
    img_root_path = '/backup/VisualFLS/imgs'
    visual_mask_path = '/backup/VisualFLS/view_mask'
    # img_files = [f for f in os.listdir(img_root_path) if f.endswith('.jpg')]
    img_files = open('/backup/VisualFLS/val.txt', 'r', encoding='utf-8').read().strip().split('\n')
    for img_name in img_files:
        img = cv2.imread(os.path.join(img_root_path, img_name), cv2.IMREAD_COLOR)
        dst = SegInfere.visual(img)
        cv2.imwrite(os.path.join(visual_mask_path, img_name), dst)
        # plt.imshow(dst)
        # plt.show()
