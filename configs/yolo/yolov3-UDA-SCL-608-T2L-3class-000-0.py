_base_ = [  './yolov3-UDA-SCL-640-L2T-3class-000-0.py', ]

samples_per_gpu=16

import os
trainpath = os.getcwd()
root_dir = trainpath[:trainpath.find('yebh')]+'yebh/'
# load_from = root_dir + '/checkpoint/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
load_from='/home/yebh/mmdet2/work_dirs/baseline/yolov3-SCL-640-T-3class-000-0-4gpu-dist/epoch_154_new.pth'
tag = '3class'
data = dict(
    samples_per_gpu=samples_per_gpu,
    train=dict( 
        source=dict(
            img_prefix=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/JPEGImages',
            ann_file=root_dir+f'dataset/BIS/BIS_HP_transparent_{tag}/HP_transparent_all_{tag}.json'),
        target=dict(
            img_prefix=root_dir+f'dataset/BIS/BIS_HP_13_16_{tag}/JPEGImages',
            ann_file=root_dir+f'dataset/BIS/BIS_HP_13_16_{tag}/HP_13_16_trainval_{tag}.json'),
        ),
    val=dict(
        img_prefix=root_dir+f'dataset/BIS/BIS_HP_13_16_{tag}/JPEGImages',
        ann_file=root_dir+f'dataset/BIS/BIS_HP_13_16_{tag}/HP_13_16_test_{tag}.json'),
    test=dict(
        img_prefix=root_dir+f'dataset/BIS/BIS_HP_13_16_{tag}/JPEGImages',
        ann_file=root_dir+f'dataset/BIS/BIS_HP_13_16_{tag}/HP_13_16_test_{tag}.json')
    )

