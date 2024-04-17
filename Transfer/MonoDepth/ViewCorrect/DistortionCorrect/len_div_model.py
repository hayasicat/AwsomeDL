# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 10:51
# @Author  : ljq
# @desc    : 
# @File    : DivModel.py
import json
import os.path

import cv2
import numpy as np

class DivModelEval():
    def __init__(self, img_shape=[0, 0]):
        self.p = [0, 0]
        self.k = [0, 0]
        self.center_points = [img_shape[1] / 2, img_shape[0] / 2]
        self.img_size = [img_shape[1], img_shape[0]]
        self.r_max = get_rmax(self.center_points, self.img_size[0], self.img_size[1])

    def set_center_points(self, center_points):
        similar_frac = []
        for i in center_points:
            similar_frac.append(round(i, 6))
        self.center_points = similar_frac
        self.r_max = get_rmax(self.center_points, self.img_size[0], self.img_size[1])

    def update_distortion_parameter(self, d, is_p=True, **kwargs):
        """
        p <-> k
        :return:
        """
        # An Iterative Optimization Algorithm for Lens Distortion Correction Using Two-Parameter Models
        similar_frac = []
        for i in d:
            similar_frac.append(np.round(i, 6))
        if is_p:
            self.p = similar_frac
            self.recompute_k()
        else:
            self.k = similar_frac
            self.recompute_p()

    def recompute_k(self):
        """
        重新更新k值
        :return:
        """
        half_r_max = self.r_max / 2
        p1 = self.p[0]
        p2 = self.p[1]
        self.k[0] = (-p1 / (1 + p1) + 16 * p2 / (1 + p2)) / (-12 * half_r_max ** 2)
        self.k[1] = (-4 * p2 / (1 + p2) + p1 / (1 + p1)) / (-12 * half_r_max ** 4)

    def recompute_p(self):
        """
        重新更新p值
        :return:
        """
        half_r_max = self.r_max / 2
        k1 = self.k[0]
        k2 = self.k[1]
        self.p[0] = 1 / (1 + k1 * self.r_max ** 2 + k2 * self.r_max ** 4) - 1
        self.p[1] = 1 / (1 + k1 * half_r_max ** 2 + k2 * half_r_max ** 4) - 1

    def set_single_parameter(self, d, is_tans_two_para=False):
        self.p[0] = d
        self.p[1] = 0
        self.k[0] = -d / (self.r_max ** 2 * (1 + d))
        self.k[1] = 0
        if is_tans_two_para:
            # 更新一下两参数的值
            self.recompute_p()
            self.recompute_k()

    def compute_scale(self, fit_type='w', keep_region_rate=1):
        # fit width
        if fit_type == 'h':
            img_points = np.array([[self.center_points[0], 0], [self.center_points[0], self.img_size[1]]])
        elif fit_type == 'all':
            img_points = np.array([[self.img_size[0], self.img_size[1] ], [0, 0], [0, self.img_size[1]],
                                   [self.img_size[0], 0]])
        else:
            img_points = np.array([[(1 - keep_region_rate) * self.center_points[0], self.center_points[1]],
                                   [self.img_size[0] - (1 - keep_region_rate) * (
                                           self.img_size[0] - self.center_points[0]), self.center_points[1]]])
        d = np.sum(np.square(img_points - self.center_points), axis=1)
        idx = np.argmax(d)
        farther_points_u = self.evaluation(img_points[idx][0], img_points[idx][1])
        r_max_u = np.sqrt(np.sum(np.square(np.array(farther_points_u) - np.array(self.center_points))))
        # 找到两边的端点做缩放判定
        end_points = np.array([[0, self.center_points[1]],
                               [self.img_size[0], self.center_points[1]]])
        r_max = np.sqrt(np.sum(np.square(end_points - self.center_points), axis=1))[idx]

        # 找到最远的r_max
        corner_points = np.array([[self.img_size[0], self.img_size[1]], [0, 0], [0, self.img_size[1]],
                                  [self.img_size[0], 0]])
        idx = np.argmax(np.sum(np.square(corner_points - np.array(self.center_points)), axis=1))
        farther_points_corner = self.evaluation(corner_points[idx][0], corner_points[idx][1])
        cor_r_max = np.sqrt(np.sum(np.square(np.array(farther_points_corner) - np.array(self.center_points))))
        return r_max_u / r_max, cor_r_max

    def gain_remap_matrix(self, fit_type='w', keep_region_rate=1):
        X, Y = np.meshgrid(range(self.img_size[0]), range(self.img_size[1]))
        scale, r_max_u = self.compute_scale(fit_type, keep_region_rate=keep_region_rate)
        inverse_vector = self.get_inverse_vector(r_max_u, self.k[0], self.k[1])
        inverse_vector = np.array(inverse_vector)
        # shift the coordinate to minus
        shift_coordinate = float(scale - 1) * np.array(self.center_points)
        X = X * scale - shift_coordinate[0]
        Y = Y * scale - shift_coordinate[1]
        ru_matrix = np.sqrt(np.square(X - self.center_points[0]) + np.square(Y - self.center_points[1]))
        int_ru = np.floor(ru_matrix).astype(np.int)
        plus_int_ru = int_ru + 1
        dec_ru = ru_matrix - int_ru
        inverse_int_ru = inverse_vector[int_ru]
        interval_ru = inverse_vector[plus_int_ru] - inverse_int_ru
        coefficient = inverse_vector[int_ru] + dec_ru * interval_ru
        XU = (X - self.center_points[0]) * coefficient + self.center_points[0]
        YU = (Y - self.center_points[1]) * coefficient + self.center_points[1]
        return XU.astype(np.float16), YU.astype(np.float16)

    def get_inverse_vector(self, r_max, k1, k2):
        r_max = int(r_max + 3)
        inverse_vector = [1]
        last_root = 1
        for i in range(1, r_max):
            rs = np.roots([k2 * i, 0, k1 * i, -1, i])
            min_idx = -1
            min_dis = 1000000000
            for idx, r in enumerate(rs):
                current_dis = 1000000000000
                if np.isreal(r) and r > 0:
                    current_dis = r - last_root
                if current_dis < min_dis:
                    min_idx = idx
                    min_dis = current_dis
            if min_idx == -1:
                raise ValueError("when r is {} has not real root".format(i))
            r = rs[min_idx]
            inverse_vector.append(r.real / i)
        return inverse_vector

    def evaluation(self, x, y):
        point = np.array([x, y])
        r2 = np.sum(np.square(point - np.array(self.center_points)))
        coefficient = 1 / (1 + r2 * self.k[0] + r2 ** 2 * self.k[1])
        result = (point - np.array(self.center_points)) * coefficient + np.array(self.center_points)
        return result

    def evaluation_line_list(self, line):
        new_line = []
        for p in line:
            result = self.evaluation(p[0], p[1])
            new_line.append(list(result))
        return new_line

    def img_inverse_distortion(self, img, fit_type='w', keep_region_rate=1):
        """
        针对图片进行反变化
        :param img:
        :return:
        """
        XU, YU = self.gain_remap_matrix(fit_type, keep_region_rate=keep_region_rate)
        mapped_img = cv2.remap(img, XU.astype(np.float32), YU.astype(np.float32), cv2.INTER_LINEAR)
        return mapped_img

    def export2file(self, file_names, fit_type='w', keep_region_rate=1):
        root_path = os.path.split(file_names)[0]
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        remap_matrix, ldm_info = self.gain_ldm_parameter(fit_type='w', keep_region_rate=keep_region_rate)
        np.save(file_names + '.npy', remap_matrix)
        # 保存畸变模型的信息
        js_str = json.dumps(ldm_info)
        with open(file_names + '.json', 'w', encoding='utf-8') as f:
            f.write(js_str)
            f.flush()

    def gain_ldm_parameter(self, fit_type='w', keep_region_rate=1):
        XU, YU = self.gain_remap_matrix(fit_type=fit_type, keep_region_rate=keep_region_rate)
        XU = XU.reshape(-1)
        YU = YU.reshape(-1)
        remap_matrix = np.stack([XU, YU], axis=1)
        ldm_info = {
            'p': [float(self.p[0]), float(self.p[1])],
            'center_points': self.center_points,
            'img_size': self.img_size,
        }
        return remap_matrix, ldm_info

    def check_remap_matrix(self,file_names):
        # 获取remap的矩阵,硬编码在data的ldm下面
        # npy file
        npy_path = file_names+'.npy'
        setting_path = file_names+'.json'
        # 对比时间，如果npy生成的时间早于setting就先生成一个新的npy 
        if os.stat(setting_path).st_mtime>os.stat(npy_path):
            XU, YU = self.gain_remap_matrix(fit_type='w', keep_region_rate=1)
            XU = XU.reshape(-1)
            YU = YU.reshape(-1)
            remap_matrix = np.stack([XU, YU], axis=1)
            np.save(npy_path, remap_matrix)
        else:
            remap_matrix = np.load(npy_path)
        return remap_matrix

    def load_from_file(self, file_names):
        js_dict = json.loads(open(file_names, 'r', encoding='utf-8').read())
        self.load_from_dict(js_dict)

    def load_from_dict(self, js_dict):
        self.img_size = js_dict.get('img_size', [0, 0])
        self.set_center_points(js_dict.get('center_points', [0, 0]))
        self.update_distortion_parameter(np.array(js_dict.get('p', [0, 0])))

    def show(self):
        print('center_point:{} \n p:{} \n img_size:{}'.format(self.center_points, self.p, self.img_size))


def get_rmax(center_point, img_w, img_h):
    c = np.array(center_point)
    img_size = np.array([[img_w - 1, img_h - 1], [0, 0], [0, img_h - 1], [img_w - 1, 0]])
    # 计算哪个是最远的边缘点
    d = np.sum(np.square(img_size - c), axis=1)
    r_max = np.sqrt(np.max(d))
    return r_max


def LrFunc(r_d, k1, k2, r_u):
    """
    return r_d zero points
    """
    return r_d / (1 + k1 * r_d ** 2 + k2 * r_d ** 4) - r_u
