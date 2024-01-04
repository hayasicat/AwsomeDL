# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 14:43
# @Author  : ljq
# @desc    : 
# @File    : test4minist.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt

import numpy as np
import matplotlib.pyplot as plt

from VisualTask.CLS import CLSTrainer
from Base.BackBone import ResNet18, SimpleCnn, STN

train_transformer = tt.Compose([
    tt.ToTensor(),
    tt.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='../data', download=True, transform=train_transformer)
test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=train_transformer)

save_path = '../data/mnist'


# 组装一下
class StackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Stn = STN(SimpleCnn, 1)
        self.backbone = ResNet18(10, input_chans=1)
        self.backbone.init_weights()

    def forward(self, x):
        x = self.Stn(x)
        return self.backbone(x)


model = StackModel()
# model = ResNet18(10, input_chans=1)
# model.init_weights()
# trainner = CLSTrainer(train_dataset, test_dataset, model, save_path=save_path)
# trainner.train()

# 载入模型权重参数，可视化stn层
model.load_state_dict(torch.load(os.path.join(save_path, '35_model.pth')))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
# device = torch.device("cuda:0")
# model.to(device)


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.Stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

#
# visualize_stn()
# plt.show()
