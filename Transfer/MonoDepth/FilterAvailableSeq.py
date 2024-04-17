# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 9:36
# @Author  : ljq
# @desc    : 
# @File    : FilterAvailableSeq.py
import os
import json


def filter_seq(path, score=0.08):
    loss_score = json.loads(open(path, 'r').read())
    available_seq = {}
    for sub_fold, s in loss_score.items():

        if s != 'nan' and eval(s) < score:
            available_seq[sub_fold] = eval(s)
    return available_seq


def convert2label(available_seq, save_path):
    f = open(save_path, 'w', encoding='utf-8')
    for subfold, s in available_seq.items():
        f.write(subfold)
        f.write('\n')
    f.flush()
    f.close()


if __name__ == "__main__":
    js_path = r'/root/project/AwsomeDL/data/sub_fold/loss_seq.json'
    result_dict = filter_seq(js_path)
    label_path = r'/root/data/BowlingMono/fragments/train.txt'
    # 保存
    convert2label(result_dict, label_path)
