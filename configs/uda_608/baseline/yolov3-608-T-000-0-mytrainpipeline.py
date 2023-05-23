_base_ = [  './yolov3-608-L-000-0-mytrainpipeline.py', ]

import os
trainpath = os.getcwd()
root_dir = trainpath[:trainpath.find('yebh')]+'yebh/'

data = dict(
    train=dict(
        img_prefix=root_dir+'dataset/BIS/BIS_HP_transparent/JPEGImages',
        ann_file=root_dir+'dataset/BIS/BIS_HP_transparent/HP_transparent_trainval.json',
        ),
    val=dict(
        img_prefix=root_dir+'dataset/BIS/BIS_HP_transparent/JPEGImages',
        ann_file=root_dir+'dataset/BIS/BIS_HP_transparent/HP_transparent_test.json',
        ),
    test=dict(
        img_prefix=root_dir+'dataset/BIS/BIS_HP_transparent/JPEGImages',
        ann_file=root_dir+'dataset/BIS/BIS_HP_transparent/HP_transparent_test.json',
        ))
