# -*- coding: utf-8 -*-
# @Time    : 2023/11/20 9:08
# @Author  : ljq
# @desc    :  分割的色彩设定：https://zhuanlan.zhihu.com/p/73134668  ;https://zhuanlan.zhihu.com/p/102303256
#             渐变色： https://blog.csdn.net/a9073b/article/details/121462333 底色从上述分割色彩开始
# @File    : SegViewer.py
import cv2
import numpy as np

from .BaseView import BaseViewer


class SegViewer(BaseViewer):
    """
    区域应该是排他的，如果是一个softmax的话显示也不会很好。输入的gt图片应该是一个H*W*1大小
    可视化kp的话
    """
    color_ = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [255, 255, 255]
    ], np.uint8)

    def __init__(self):
        super().__init__()

    def cls_view(self, img, pred, gt=None, color=None, **kwargs):
        pred = pred.astype(np.int32)
        pred[pred > 20] = 21
        if color is None:
            mask = self.color_[pred]
        else:
            mask = color[pred]
        # 添加上去
        dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        return dst

    def kp_view(self, img, pred, gt=None, **kwargs):
        # 渐变色，pred应该是HWC，其中C代表的是颜色的类别
        mask = np.zeros_like(img, np.float32)
        # 对每个通道进行处理然后再相加，如果超出255的话就截断,判断一下如果Pred是多个通道的话
        channel_num = 1
        if len(pred.shape) == 3:
            channel_num = pred.shape[-1]
        for i in range(channel_num):
            color = self.color_[i + 1].astype(np.float32)
            current_mask = np.zeros((pred.shape[0], pred.shape[1], 3), np.float32)
            for x, y in zip(*np.where(pred[:, :, i] > 0)):
                current_mask[x, y, :] = color
            current_mask = pred[:, :, i].reshape((pred.shape[0], pred.shape[1], 1)) * current_mask
            mask += current_mask
        mask[mask >= 255] = 255
        mask = mask.astype(np.uint8)
        dst = cv2.addWeighted(img, 0.3, mask, 0.7, 0)
        return dst


if __name__ == "__main__":
    # 检查的结果
    import torchvision
    import matplotlib.pyplot as plt

    train_dataset = torchvision.datasets.voc.VOCSegmentation('../../data')
    for i in range(1):
        img, seg = train_dataset[i]
        img = np.array(img)
        seg = np.array(seg)
        V = SegViewer()
        res = V.cls_view(img, seg)
        plt.imshow(res)
        plt.show()

    from Transfer.VisualFLS.dataset import FLSHybridDataset

    dataset = FLSHybridDataset('/backup/VisualFLS', is_crop=False)
    for i in range(1):
        img, seg, heatmap = dataset[i]
        img = img.transpose(1, 2, 0)
        heatmap = heatmap.transpose(1, 2, 0)
        res = V.kp_view(img, heatmap)
        plt.imshow(res)
        plt.show()
