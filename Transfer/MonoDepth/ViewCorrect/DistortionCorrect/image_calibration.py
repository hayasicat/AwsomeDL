#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def order_points(pts):
    """
    Order points start from top-left in clock-wise order
    Args: unordered points
        pts(np.array, [N, 2])

    Returns:
        rect(np.array, [4, 2]): order points of nx2 to 4x2 of order top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(image, pts, x_border=0, y_border=0):
    """Perform perspective transform of an image
    Args:
        image(np.array, [H, W, 3]):
        pts(np.array, [N, 2]):
        x_border(int): the border space beyond target polygon along x axis
        y_border(int): the border space beyond target polygon along y axis

    Returns:
        warped(np.array, [H', W', 3]): perspective transformed image
        M(np.array, [3, 3]): transform matrix
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # define our dst using x_border and y_border
    dst = np.array([
        [x_border, y_border],
        [maxWidth + x_border, y_border],
        [maxWidth + x_border, maxHeight + y_border],
        [x_border, maxHeight + y_border]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth + x_border * 2, maxHeight + y_border * 2))
    return warped, M


def image_precess(image, bounding_box,frame_zoom_value, zoom_y, **kwargs):
    # [矩形四个顶点， X轴边界扩充值，Y轴边界扩充值，裁剪值，旋转角度，缩放值， Y周缩放值]
    # 缩放图像
    image = cv2.resize(image, None, fx=frame_zoom_value, fy=frame_zoom_value * zoom_y, interpolation=cv2.INTER_CUBIC)
    height, width = image.shape[:2]
    # 裁剪图像
    Bounding_ymin = max(int(bounding_box[0] * zoom_y), 0)
    Bounding_ymax = min(int(bounding_box[1] * zoom_y), height)
    Bounding_xmin = max(bounding_box[2], 0)
    Bounding_xmax = min(bounding_box[3], width)
    image = image[Bounding_ymin:Bounding_ymax, Bounding_xmin:Bounding_xmax]

    return image


if __name__ == '__main__':
    img = np.zeros((3840, 2160, 3), np.uint8)
    Four_Points = [[1301, 876], [1321, 1483], [2707, 1457], [2726, 863]]
    Border_X = 1500
    Border_Y = 700
    Bounding_Box = [0, 1174, 0, 2213]
    Rotate_Angle = 0
    frame_zoom_value = 0.5
    Zoom_Y = 1.3
