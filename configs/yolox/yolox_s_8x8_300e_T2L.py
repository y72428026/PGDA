_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    bbox_head=dict(in_channels=128, feat_channels=128, num_classes=6)
    )

dataset_type = 'COCODataset'
classes = (
    'triangle offset' ,
    'leftover material',
    'left and right side material',
    'open on both sides',
    'big hole opening',
    'white border')

data = dict(
    samples_per_gpu=12, # 8*1.5
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
            classes=classes,
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_all.json')),
    val=dict(
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
            classes=classes,
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json'),
    test=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json'),
    )

optimizer = dict(
    type='SGD',
    lr=0.015, #0.01*1.5
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

