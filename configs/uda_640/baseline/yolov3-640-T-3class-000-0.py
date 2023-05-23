_base_ = [  './yolov3-640-L-3class-000-0.py', ]

import os
trainpath = os.getcwd()
root_dir = trainpath[:trainpath.find('yebh')]+'yebh/'
tag = '3class'
data = dict(
    train=dict(
        img_prefix=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/JPEGImages',
        ann_file=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/HP_transparent_trainval_{tag}.json',
        ),
    val=dict(
        img_prefix=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/JPEGImages',
        ann_file=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/HP_transparent_test_{tag}.json',
        ),
    test=dict(
        img_prefix=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/JPEGImages',
        ann_file=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/HP_transparent_test_{tag}.json',
        ))
