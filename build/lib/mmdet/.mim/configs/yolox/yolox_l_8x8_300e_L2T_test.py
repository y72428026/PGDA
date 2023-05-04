_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

dataset_type = 'CocoDataset_DA'
classes = (
    'triangle offset' ,
    'leftover material',
    'left and right side material',
    'open on both sides',
    'big hole opening',
    'white border')

img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset_DA',
    dataset=dict(
        type='CocoDataset_DA',
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_L2T/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_L2T/HP_L2T_trainval.json',
        num_src= 1517,
        num_trg= 1346,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False
    ),
    pipeline=train_pipeline,
)

data = dict(
    train=train_dataset,
    val=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_L2T/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_L2T/HP_L2T_test.json'),
    test=dict(
        img_prefix='/media/data3/yebh/dataset/BIS/HP/BIS_HP_L2T/JPEGImages',
        classes=classes,
        ann_file='/media/data3/yebh/dataset/BIS/HP/BIS_HP_L2T/HP_L2T_test.json'))

