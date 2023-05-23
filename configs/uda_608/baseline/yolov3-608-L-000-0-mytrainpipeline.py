_base_ = [  '../../yolo/yolov3_d53_mstrain-608_273e_coco.py', ]

import os
trainpath = os.getcwd()
root_dir = trainpath[:trainpath.find('yebh')]+'yebh/'

samples_per_gpu=32
# evaluation = dict(interval=1, metric=['bbox'], start=30)
evaluation = dict(interval=1, metric=['bbox'])
log_config = dict(interval=30)
find_unused_parameters=True
load_from = root_dir + '/checkpoint/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
model = dict(
    bbox_head=dict(
        num_classes=6,
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            # base_sizes=[[(116, 90), (156, 198), (373, 326)],
            #             [(30, 61), (62, 45), (59, 119)],
            #             [(10, 13), (16, 30), (33, 23)]],
            base_sizes=[[(439, 110), (220, 243), (419, 153)],
                        [(135, 104), (144, 189), (334, 94)],
                        [(31, 28), (33, 123), (70, 61)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum'))
    )

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_type = 'CocoDataset_mine'

classes = (
    'triangle offset' ,
    'leftover material',
    'left and right side material',
    'open on both sides',
    'big hole opening',
    'white border')
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=root_dir+'dataset/BIS/BIS_HP_13_16/JPEGImages',
        ann_file=root_dir+'dataset/BIS/BIS_HP_13_16/HP_13_16_trainval.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=root_dir+'dataset/BIS/BIS_HP_13_16/JPEGImages',
        ann_file=root_dir+'dataset/BIS/BIS_HP_13_16/HP_13_16_test.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=root_dir+'dataset/BIS/BIS_HP_13_16/JPEGImages',
        ann_file=root_dir+'dataset/BIS/BIS_HP_13_16/HP_13_16_test.json',
        pipeline=test_pipeline))

cudnn_benchmark = True

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[218, 246])

runner = dict(type='EpochBasedRunner', max_epochs=273)
auto_scale_lr = dict(base_batch_size=64)