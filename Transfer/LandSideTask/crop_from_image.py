# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 10:31
# @Author  : ljq
# @desc    : 
# @File    : crop_from_image.py
from shapely.geometry import Polygon


def get_overlap_coordinates(data1, data2):
    poly1 = Polygon(data1)
    poly2 = Polygon(data2)
    if not poly1.intersects(poly2):
        return False
    else:
        overlap = poly1.intersection(poly2)
        return True, list(overlap.exterior.coords)


# 传入两个多边形的顶点坐标 data1 和 data2
data1 = [(0, 0), (0, 2), (2, 2), (2, 0)]
data2 = [(1, 1), (1, 3), (3, 3), (3, 1)]

overlap_coordinates = get_overlap_coordinates(data1, data2)
if overlap_coordinates:
    print(f"重叠区域的坐标为: {overlap_coordinates}")
else:
    print("两个多边形不相交，没有重叠区域。")
