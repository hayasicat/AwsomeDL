# -*- coding: utf-8 -*-
# @Time    : 2024/2/26 10:36
# @Author  : ljq
# @desc    : 
# @File    : crop_image.py
import os
import cv2
import matplotlib.pyplot as plt
from Transfer.VisualFLS.DataProcess.TCProcessor import TCBase, TCProcessor

img_root = r'G:\fls_ann\reserve\reserve'
img_files = [i for i in os.listdir(img_root) if i.endswith('.jpg')]
p = TCProcessor()
for idx, img_name in enumerate(img_files):
    img_path = os.path.join(img_root, img_name)
    print(idx, img_path)
    result = p.transform(img_path)
    img = result['img_patch']
    plt.imshow(img)
    plt.show()
