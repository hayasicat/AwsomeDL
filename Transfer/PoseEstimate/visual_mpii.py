# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 17:14
# @Author  : ljq
# @desc    : 
# @File    : visual_mpii.py
import matplotlib.pyplot as plt

from Base.Dataset.mpii import MPII

dataset = MPII(r'/backup/PoseEstimate/MPII/mpii-hr-lsp-normalizer.json', r'/backup/PoseEstimate/MPII/images')
for i in range(2):
    sample = dataset[i]
    heatmap = sample[1]
    print(heatmap.size())
    # for n in range(heatmap.size()[0]):
    #     plt.imshow(heatmap[n, :, :].numpy())
    #     plt.show()
