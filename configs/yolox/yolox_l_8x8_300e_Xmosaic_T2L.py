_base_ = './yolox_s_8x8_300e_coco_Xmosaic.py'

img_scale = (640, 480)
# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256, num_classes=6))

dataset_type = 'COCODataset'
classes = (
    'triangle offset' ,
    'leftover material',
    'left and right side material',
    'open on both sides',
    'big hole opening',
    'white border')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_all.json'),
    val=dict(
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
            classes=classes,
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_test.json'),
    test=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_test.json'),
    )
auto_scale_lr = dict(base_batch_size=64)



