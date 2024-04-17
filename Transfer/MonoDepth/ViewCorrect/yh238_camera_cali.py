import os
import cv2
import numpy as np
from Transfer.MonoDepth.ViewCorrect.CamCali import SingleCamera
from Transfer.MonoDepth.ViewCorrect.DistortionCorrect import DivModelEval

setting_path = r'D:\project\ReBowling\data\YH238\cam_front.json'
img_root = r'D:\project\ReBowling\data\YH238\20230817\front'
front_path = os.path.join(img_root, 'source')
undistortion_path = os.path.join(img_root, 'undistortion')
bev_path = os.path.join(img_root, "bev_correct")
bev_npy_path = os.path.join(img_root, "bev.npy")

ldm = DivModelEval()
ldm.load_from_file(setting_path)
ldm.show()

world_points = [
    [12192, 2438, -2896, 1], [12192, 0, -2896, 1], [0, 0, -2896, 1], [0, 2438, -2896, 1],
    [12192, 2438, -5792, 1], [12192, 0, -5792, 1], [0, 0, -5792, 1], [0, 2438, -5792, 1],
    [12192, 2438, -8688, 1], [12192, 0, -8688, 1], [0, 0, -8688, 1], [0, 2438, -8688, 1],
    [12192, 2438, -11584, 1], [12192, 0, -11584, 1], [0, 0, -11584, 1], [0, 2438, -11584, 1],
    [12192, 2438, -14480, 1], [12192, 0, -14480, 1], [0, 0, -14480, 1], [0, 2438, -14480, 1],
]

image_points = [
    [1222, 693], [1223, 575], [640, 557], [634, 676],
    [1266, 707], [1266, 570], [588, 551], [581, 688],
    [1329, 727], [1330, 562], [520, 539], [509, 702],
    [1421, 757], [1421, 554], [414, 525], [399, 727],
    [1571, 809], [1565, 538], [241, 497], [217, 766],
]


def correcte_distortion(img):
    img = ldm.img_inverse_distortion(img)
    return img


def get_camera_matrix(world_loc, pixel_loc):
    world_points = np.array(world_loc).reshape((-1, 4))
    image_points = np.array(pixel_loc).reshape((-1, 2))
    min_average_err = 10000
    # 循环查找最小值
    min_n = 10000
    parameters = []
    for n in range(13):
        input_world_points = world_points[n:n + 6]
        input_image_points = image_points[n:n + 6]

        check_world_points = world_points
        check_image_points = image_points
        s = SingleCamera(input_world_points, input_image_points, 12)
        s.composeP()
        s.svdP()
        s.workInAndOut()
        average_err = s.selfcheck(check_world_points, check_image_points)
        parameters.append(s)
        if average_err < min_average_err:
            min_average_err = average_err
            min_n = n
    parameter = parameters[3]
    # 
    return parameter._K, parameter._R, parameter._t


class Transform2BEV():
    def __init__(self, K, R, T, img_shape) -> None:
        # 系数间相乘
        K_inv = np.linalg.inv(K)
        R_T = R.T
        # 系数和偏移量
        coeff = K @ R @ K_inv
        devia = K @ R @ T + K @ T
        # 生成新图片矩阵
        X, Y = np.meshgrid(range(img_shape[1]), range(img_shape[0]))
        ones_mat = np.ones_like(X)
        new_image_coordinate = np.stack([X, Y, ones_mat])
        new_image_coordinate = new_image_coordinate.reshape(
            (new_image_coordinate.shape[0], new_image_coordinate.shape[1] * new_image_coordinate.shape[2]))
        new_image_coordinate = coeff @ new_image_coordinate
        new_image_coordinate = new_image_coordinate.reshape((new_image_coordinate.shape[0], img_shape[0], img_shape[1]))
        self.normal_x = new_image_coordinate[0, :, :] / new_image_coordinate[2, :, :]
        self.normal_y = new_image_coordinate[1, :, :] / new_image_coordinate[2, :, :]

    def transform(self, img, is_crop=False, bbox=[0, 10000, 0, 30000]):
        mapped_img = cv2.remap(img, self.normal_x.astype(np.float32), self.normal_y.astype(np.float32),
                               cv2.INTER_LINEAR)
        # crops
        if is_crop:
            img_shape = mapped_img.shape[:2]
            x_max = img_shape[1]
            y_max = img_shape[0]
            # 限制长宽高都要为32的整数倍
            new_bbox = [max(0, bbox[0]), min(x_max, bbox[1]), max(0, bbox[2]), min(y_max, bbox[3])]
            new_bbox[1] = int((new_bbox[1] - new_bbox[0]) / 32) * 32 + new_bbox[0]
            new_bbox[3] = int((new_bbox[3] - new_bbox[2]) / 32) * 32 + new_bbox[2]
            mapped_img = mapped_img[new_bbox[2]:new_bbox[3], new_bbox[0]:new_bbox[1], :]
        return mapped_img

    def save_bev_matrix(self):
        remap_matrix = np.stack([self.normal_x.astype(np.float16), self.normal_y.astype(np.float16)], axis=2)
        np.save(bev_npy_path, remap_matrix)


if __name__ == "__main__":
    is_save_undistortion = False
    is_save_bev = False
    img_names = os.listdir(front_path)
    K, R, T = get_camera_matrix(world_points, image_points)
    # 
    bev_transformer = Transform2BEV(K, R, T, [1080, 1920])
    print(K, R, T)
    for img_name in img_names:
        img_path = os.path.join(front_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        img = correcte_distortion(img)
        if is_save_undistortion:
            if not os.path.exists(undistortion_path):
                os.makedirs(undistortion_path)
            cv2.imwrite(os.path.join(undistortion_path, img_name), img)
        # 裁剪以后内参会发生变化，尝试直接使用内参训练和改变内参的方式训练
        mapping_img = bev_transformer.transform(img, is_crop=True, bbox=[45, 1900, 25, 883])
        cv2.imshow("img", mapping_img)
        cv2.waitKey(0)
        # break
        if is_save_bev:
            if not os.path.exists(bev_path):
                os.makedirs(bev_path)
            # cv2.imwrite(os.path.join(bev_path,img_name),mapping_img)
    if is_save_bev:
        bev_transformer.save_bev_matrix()
        # break

# ----------------------
# K
# [[972.63078965   4.4270343  955.42699351]
#  [  0.         972.81690651 534.1624363 ]
#  [  0.           0.           1.        ]]
# R
# [[ 0.99960112 -0.02785725 -0.00464437]
#  [ 0.02812252  0.99692451  0.07314815]
#  [ 0.00259238 -0.07324958  0.99731027]]
# T
# [[-6628.19953849]
# [  725.82146012]
# [23247.36837053]]
