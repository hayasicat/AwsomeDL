# -*- encoding: utf-8 -*-
# @文件        :dlt_camera_calibration.py
# @说明        :
# @时间        :2023/05/22 14:14:28
# @作者        :ljq
import numpy as np
import numpy.linalg as LA


class SingleCamera:

    def __init__(self, world_coor, pixel_coor, n):

        self.__world_coor = world_coor
        self.__pixel_coor = pixel_coor
        self.__point_num = n

        '''
        0. P is the appropriate form when Pm=0
        1. SVD-solved M is known up to scale, 
        which means that the true values of the camera matrix are some scalar multiple of M,
        recorded as __roM
        2. __M can be represented as form [A b], where A is a 3x3 matrix and b is with shape 3x1
        3. __K is the intrisic Camera Matrix  
        4. __R and __t for rotation and translation
        
        '''
        self.__P = np.empty([self.__point_num, 12], dtype=float)
        self.__roM = np.empty([3, 4], dtype=float)
        self.__A = np.empty([3, 3], dtype=float)
        self.__b = np.empty([3, 1], dtype=float)
        self.__K = np.empty([3, 3], dtype=float)
        self.__R = np.empty([3, 3], dtype=float)
        self.__t = np.empty([3, 1], dtype=float)

    def returnAb(self):
        return self.__A, self.__b

    def returnKRT(self):
        return self.__K, self.__R, self.__t

    def returnM(self):
        return self.__roM

    def myReadFile(filePath):
        pass

    def changeHomo(no_homo):
        pass

    # to compose P in right form s.t. we can get Pm=0
    def composeP(self):
        i = 0
        P = np.empty([self.__point_num, 12], dtype=float)
        # print(P.shape)
        while i < self.__point_num:
            c = i // 2
            p1 = self.__world_coor[c]
            p2 = np.array([0, 0, 0, 0])
            if i % 2 == 0:
                p3 = -p1 * self.__pixel_coor[c][0]
                # print(p3)
                P[i] = np.hstack((p1, p2, p3))

            elif i % 2 == 1:
                p3 = -p1 * self.__pixel_coor[c][1]
                # print(p3)
                P[i] = np.hstack((p2, p1, p3))
            # M = P[i]
            # print(M)
            i = i + 1
        print("Now P is with form of :")
        print(P)
        print('\n')
        self.__P = P

    # svd to P，return A,b, where M=[A b]
    def svdP(self):
        # self.__P = self.__P[:self.__point_num,:]
        # print(self.__P)
        U, sigma, VT = LA.svd(self.__P)
        V = np.transpose(VT)
        preM = V[:, -1]
        roM = preM.reshape(3, 4)
        print("some scalar multiple of M,recorded as roM:")
        print(roM)
        print('\n')
        A = roM[0:3, 0:3].copy()
        b = roM[0:3, 3:4].copy()
        print("M can be written in form of [A b], where A is 3x3 and b is 3x1, as following:")
        print(A)
        print(b)
        print('\n')
        self.__roM = roM
        self.__A = A
        self.__b = b

    # solve the intrinsics and extrisics
    def workInAndOut(self):
        # compute ro, where ro=1/|a3|, ro may be positive or negative,
        # we choose the positive ro and name it ro01
        a3T = self.__A[2]
        # print(a3T)
        under = LA.norm(a3T)
        # print(under)
        ro01 = 1 / under
        print("The ro is %f \n" % ro01)

        # comput cx and cy
        a1T = self.__A[0]
        a2T = self.__A[1]
        cx = ro01 * ro01 * (np.dot(a1T, a3T))
        cy = ro01 * ro01 * (np.dot(a2T, a3T))
        print("cx=%f,cy=%f \n" % (cx, cy))

        # compute theta
        a_cross13 = np.cross(a1T, a3T)
        a_cross23 = np.cross(a2T, a3T)
        theta = np.arccos((-1) * np.dot(a_cross13, a_cross23) / (LA.norm(a_cross13) * LA.norm(a_cross23)))
        print("theta is: %f \n" % theta)

        # compute alpha and beta
        alpha = ro01 * ro01 * LA.norm(a_cross13) * np.sin(theta)
        beta = ro01 * ro01 * LA.norm(a_cross23) * np.sin(theta)
        print("alpha:%f, beta:%f \n" % (alpha, beta))

        # compute K
        K = np.array([alpha, -alpha * (1 / np.tan(theta)), cx, 0, beta / (np.sin(theta)), cy, 0, 0, 1])
        K = K.reshape(3, 3)
        print("We can get K accordingly: ")
        print(K)
        print('\n')
        self._K = K

        # compute R
        r1 = a_cross23 / LA.norm(a_cross23)
        r301 = ro01 * a3T
        r2 = np.cross(r301, r1)
        # print(r1, r2, r301)
        R = np.hstack((r1, r2, r301))
        R = R.reshape(3, 3)
        print("we can get R:")
        print(R)
        print('\n')
        self._R = R

        # compute T
        T = ro01 * np.dot(LA.inv(K), self.__b)
        print("we can get t:")
        print(T)
        print('\n')
        self._t = T
        # TODO：增加一个非线性优化器来减小误差

    def selfcheck(self, w_check, c_check):
        my_size = c_check.shape[0]
        my_err = np.empty([my_size])
        for i in range(my_size):
            test_pix = np.dot(self.__roM, w_check[i])
            u = test_pix[0] / test_pix[2]
            v = test_pix[1] / test_pix[2]
            u_c = c_check[i][0]
            v_c = c_check[i][1]
            print("you get test point %d with result (%f,%f)" % (i, u, v))
            print("the correct result is (%f,%f)" % (u_c, v_c))
            my_err[i] = (abs(u - u_c) / u_c + abs(v - v_c) / v_c) / 2
        average_err = my_err.sum() / my_size
        print("The average error is %f ," % average_err)
        if average_err > 0.1:
            print("which is more than 0.1,error is {}".format(average_err))
        else:
            print("which is smaller than 0.1, the M is acceptable")
        return average_err


if __name__ == '__main__':
    # 计算内参和外参
    # world_points = [[0,12192,0,1],[0,12192,2591,1],[2438,12192,2591,1],
    #                 [2438,0,2591,1],[0,0,2591,1],[0,0,0,1]]
    # image_points = [[663,328],[626,299],[620,131],
    #                 [1464,101],[1468,272],[1404,302]]

    # 一层和二层
    # world_points = [[0,0,2591,1],[0,2438,2591,1],[12192,2438,2591,1],[12192,0,2591,1],
    #                 [0,0,5182,1],[0,2438,5182,1],[12192,2438,5182,1],[12192,0,5182,1]]
    # image_points = [[632,692],[628,522],[1479,492],[1483,667],
    #                 [566,729],[558,524],[1590,492],[1596,699]]
    world_points = [[2438, 0, 2591, 1], [0, 0, 2591, 1], [0, 12192, 2591, 1], [2438, 12192, 2591, 1],
                    [2438, 0, 5182, 1], [0, 0, 5182, 1], [0, 12192, 5182, 1], [2438, 12192, 5182, 1],
                    [2438, 0, 7773, 1], [0, 0, 7773, 1], [0, 12192, 7773, 1], [2438, 12192, 7773, 1],
                    [2438, 0, 10364, 1], [0, 0, 10364, 1]
                    ]
    image_points = [[632, 692], [628, 522], [1479, 492], [1483, 667],
                    [566, 729], [558, 524], [1590, 492], [1596, 699],
                    [464, 788], [454, 526], [1758, 486], [1765, 750],
                    [286, 875], [274, 520]
                    ]

    world_points = np.array(world_points).reshape((-1, 4))
    image_points = np.array(image_points).reshape((-1, 2))

    input_world_points = world_points[:6]
    input_image_points = image_points[:6]
    check_world_points = world_points[3:]
    check_image_points = image_points[3:]

    s = SingleCamera(input_world_points, input_image_points, 12)
    s.composeP()
    s.svdP()
    s.workInAndOut()
    s.selfcheck(check_world_points, check_image_points)

    # 从真实坐标映射到像素坐标
    w_coor = check_world_points[:, :3].T
    comput_c = np.matmul(s._R, w_coor)
    comput_uv = np.matmul(s._K, comput_c + s._t)
    sacle_uv = comput_uv[:, :] / comput_uv[2, :]
    print(sacle_uv, check_image_points)

    # 计算真实世界坐标公式PW = zR^{-1}K^{-1}Puv-R^{-1}T
    # 其中S的值为 Zc = Z_w+R^{-1}T[2]/R^{-1}K^{-1}P_uv[2]
    inv_K = np.linalg.inv(s._K)
    inv_R = np.linalg.inv(s._R)
    uv_coor = np.ones(check_image_points.shape[0]).reshape((-1, 1))
    uv_coor = np.hstack([check_image_points, uv_coor]).T

    # 深度信息 
    inv_rk = np.matmul(np.matmul(inv_R, inv_K), uv_coor)
    inv_t = np.matmul(inv_R, s._t)
    Zc = (check_world_points[:, 2].T + inv_t[2]) / inv_rk[2, :]
    Zc = Zc.reshape((-1, Zc.shape[0]))
    compute_w = Zc[0] * inv_rk - inv_t
    print(compute_w)

    # 自己添加另外一张图的几点来验证坐标对不对
    other_world_points = [[2438, 0, 2591, 1], [0, 0, 2591, 1]]
    other_image_points = [[1464, 101], [1468, 272]]
    other_world_points = np.array(other_world_points).reshape((-1, 4))
    other_image_points = np.array(other_image_points).reshape((-1, 2))
    inv_K = np.linalg.inv(s._K)
    inv_R = np.linalg.inv(s._R)
    uv_coor = np.ones(other_image_points.shape[0]).reshape((-1, 1))
    uv_coor = np.hstack([other_image_points, uv_coor]).T

    # 深度信息 
    inv_rk = np.matmul(np.matmul(inv_R, inv_K), uv_coor)
    inv_t = np.matmul(inv_R, s._t)
    Zc = (other_world_points[:, 2].T + inv_t[2]) / inv_rk[2, :]
    Zc = Zc.reshape((-1, Zc.shape[0]))
    compute_w = Zc[0] * inv_rk - inv_t
    print(compute_w)
