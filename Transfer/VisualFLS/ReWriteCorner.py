# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 18:44
# @Author  : ljq
# @desc    : 
# @File    : ReWriteCorner.py
import os
import json

grasp_container = {
    'c1': [503, 310],
    'c2': [455, 310],
    'c3': [446, 10],
    'c4': [497, 2],
    'rect_shape': 768
}

fold_container = {
    'c1': [597, 381],
    'c2': [871, 414],
    'c3': [587, 402],
    'c4': [943, 283],
    'rect_shape': 384
}
kp_types = ['HoistedContainerCorner', 'ContainerSurfaceCorner']
fold_and_grasp = ['LockHole', 'HoistedContainer']


def rewrite_corner_label(base_fold, target_fold):
    # 找到有角点的图片
    if not os.path.exists(target_fold):
        os.makedirs(target_fold)
    files = os.listdir(base_fold)
    js_files = [f_name for f_name in files if f_name.endswith('.json')]
    # 重新定位角点的位置
    for js_name in js_files:
        current_kp = {}
        anns = json.loads(open(os.path.join(base_fold, js_name), 'r', encoding='utf-8').read())['shapes']
        is_fold = False
        for ann in anns:
            if ann['label'] in kp_types:
                current_kp[ann['label']] = ann['points']
            if ann['label'] == 'HoistedContainerCorner':
                is_fold = True

        if len(list(current_kp.keys())) == 0:
            # 这张图里面有没角点，可以跳过
            continue
        channel_name = js_name.split('_')[-2]
        if is_fold:
            left_top = fold_container[channel_name]
        else:
            left_top = grasp_container[channel_name]
        # 重新计算新的图片坐标
        for type_name, type_value in current_kp.items():
            for idx, num in enumerate(left_top):
                type_value[0][idx] -= num
        # 重新写入
        write_path = os.path.join(target_fold, js_name)
        with open(write_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(current_kp))
            f.flush()


if __name__ == '__main__':
    rewrite_corner_label(r'/backup/VisualFLS/imgs', '/backup/VisualFLS/seg')
