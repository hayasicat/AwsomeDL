# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 11:41
# @Author  : ljq
# @desc    : 
# @File    : coordinates_overlap.py
from shapely.geometry import Polygon, MultiPolygon


def get_overlap_coordinates(data1, data2):
    poly1 = Polygon(data1)
    poly2 = Polygon(data2)
    if not poly1.intersects(poly2):
        return False, [[]]
    else:
        overlap = poly1.intersection(poly2)
        if isinstance(overlap, Polygon):
            return True, [list(overlap.exterior.coords)]
        else:
            coors = []
            for p in overlap.geoms:
                coors.append(list(p.exterior.coords))
            return True, coors
