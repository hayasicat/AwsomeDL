# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 15:08
# @Author  : ljq
# @desc    : 
# @File    : dataset.py
import os
import json
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class MonoDataset(data.Dataset):
    K = np.array([[0.532, 0, 0.501, 0],
                  [0, 1.168, 0.611, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    org_height = 832
    org_width = 1824

    def __init__(self, data_root, image_file, height, width, num_scale=4, is_train=True, img_ext='.png',
                 coor_shift=[0, 0]):
        """
        image_file: subfold_path idx  -> os.path.join(data_root,subfold_path+'/'+str(idx)+img_ext)
        """
        self.frame_index = [-1, 0, 1]  # 实际上是[0,-1,1]
        self.reset_input_image_size(height, width, coor_shift[0], coor_shift[1])
        self.coor_shift = coor_shift
        self.is_train = is_train
        self.img_ext = img_ext
        self.height = height
        self.width = width
        self.num_scale = num_scale
        self.data_root = data_root

        self.filenames = self.get_filenames(image_file)
        self.resize_trans = {}
        for i in range(self.num_scale):
            s = 2 ** i
            self.resize_trans[i] = transforms.Resize((self.height // s, self.width // s))
        # 全部都转换为torch tensor
        self.to_tensor = transforms.ToTensor()
        # pseudo label
        self.has_pseudo_label = False
        pseudo_file_name = 'pseudo_pose.json'
        label_root, train_file = os.path.split(image_file)
        pseudo_path = os.path.join(label_root, pseudo_file_name)
        if os.path.exists(pseudo_path):
            # 读取label
            self.has_pseudo_label = True
            with open(pseudo_path, 'r', encoding='utf-8') as f:
                self.pseudo_info = json.loads(f.read().strip())
        self.augment = transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))

    def get_filenames(self, image_file):
        return open(image_file).readlines()

    def get_read_file(self, index):
        return self.filenames[index].strip().split(' ')

    def reset_input_image_size(self, height, width, x_shift, y_shift):
        Kx = self.K[0, :] * self.org_width
        Ky = self.K[1, :] * self.org_height
        Kx[2] = Kx[2] - x_shift
        Ky[2] = Ky[2] - y_shift
        # 进行缩放
        Kx = Kx / (self.org_width - 2 * x_shift)
        Ky = Ky / (self.org_height - 2 * y_shift)

        self.K[0, :] = Kx
        self.K[1, :] = Ky

    def get_color(self, folder, frame_index):
        image_path = os.path.join(self.data_root, folder, str(frame_index) + self.img_ext)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        end_y = img.shape[0] - self.coor_shift[1]
        end_x = img.shape[1] - self.coor_shift[0]
        img = img[self.coor_shift[1]:end_y, self.coor_shift[0]:end_x, :]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        prime代表的原图缩放到相关的尺寸，shrink代表的是缩小的层次，一共四级

        :param index:
        :return:
        """
        subfold, frame_idx = self.get_read_file(index)
        inputs = {}
        for rank in self.frame_index:
            image_name = 'prime' + str(rank) + '_0'
            img = self.get_color(subfold, eval(frame_idx) + rank)

            inputs[image_name] = np.array(self.resize_trans[0](Image.fromarray(img)))

        # 缩放一下
        for scale in range(self.num_scale):
            # 上下帧的图片也要调整
            if scale != 0:
                for rank in self.frame_index:
                    image_name = 'prime' + str(rank) + '_0'
                    img = inputs[image_name]
                    image_k = 'prime' + str(rank) + '_' + str(scale)
                    # 缩放图片以及针对内参进行处理
                    inputs[image_k] = np.array(self.resize_trans[scale](Image.fromarray(img)))

            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)
            K_name = "K_" + str(scale)
            inv_K_name = "inv_K" + str(scale)
            inputs[K_name] = K
            inputs[inv_K_name] = inv_K
        for k_name, v_ in inputs.items():
            if 'prime' in k_name:
                # inputs[k_name] = torch.from_numpy(v_).to(torch.float32).permute(2, 0, 1)
                inputs[k_name] = self.to_tensor(v_)
            else:
                inputs[k_name] = torch.from_numpy(v_)

        for rank in self.frame_index:
            image_name = 'prime' + str(rank) + '_0'
            inputs[image_name + 'aug'] = self.augment(inputs[image_name])

        # 读取位姿的伪标签,如果有的话就塞进去给个伪标签
        if self.has_pseudo_label:
            current_pose = self.pseudo_info.get(self.filenames[index].strip(), {})
            key_names = list(current_pose.keys())
            if 'pre_t' in key_names and 'next_t' in key_names:
                inputs['pre_t'] = np.array(current_pose['pre_t']).reshape(1, -1)
                inputs['next_t'] = np.array(current_pose['next_t']).reshape(1, -1)
            else:
                # 如果没有的话就塞进去一个000
                inputs['pre_t'] = np.array([0.0, 0.0, 0.0]).reshape(1, -1)
                inputs['next_t'] = np.array([0.0, 0.0, 0.0]).reshape(1, -1)
        return inputs


class MonoDatasetFold(MonoDataset):
    def __init__(self, data_root, image_file, height, width, num_scale=4, is_train=True, img_ext='.png',
                 coor_shift=[0, 0]):
        super().__init__(data_root, image_file, height, width, num_scale=num_scale, is_train=is_train, img_ext=img_ext,
                         coor_shift=coor_shift)
        self.sub_fold = image_file

    def get_filenames(self, image_file):
        sub_root = os.path.join(self.data_root, image_file)
        filenames = os.listdir(sub_root)
        # 肯定是要大于等于三个样本的
        filenames = sorted(filenames, key=lambda x: eval(x.rsplit('.')[0]))[1:-1]
        return filenames

    def get_read_file(self, index):
        return self.sub_fold, self.filenames[index].rsplit('.')[0]


if __name__ == "__main__":
    r = r'/root/project/AwsomeDL/data/BowlingMono'
    f_p = os.path.join(r, r'bowling/train_files.txt')
    d = MonoDataset(r, f_p, 416, 896, coor_shift=[16, 0])
    # d = MonoDataset(r, f_p, 832, 1824)
    # for i in range(30):
    #     o = d[i]

    d1 = MonoDatasetFold('/root/data/BowlingMono/fragments', 'newnvr238_ch8_20230803000011_20230803105251/image_68',
                         416, 896, coor_shift=[16, 0])
    for i in range(5):
        print(d1[i])
    # 测试不带txt版本的怎么训起来

    from torch.utils.data import DataLoader
    #
    # for k, v in o.items():
    #     # print(k, v.shape, v)
    #     print(1)
    # l = DataLoader(d, 2)
    # for o in l:
    #     for k, v in o.items():
    #         print(k, v.shape, v.dtype)
