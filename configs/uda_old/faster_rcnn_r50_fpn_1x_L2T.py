_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=6
        )
    )
)

dataset_type = 'CocoDataset_mine'
classes = (
    'triangle offset' ,
    'leftover material',
    'left and right side material',
    'open on both sides',
    'big hole opening',
    'white border')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale=(640,480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_test.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_test.json',
        pipeline=test_pipeline))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[218, 246])

runner = dict(type='EpochBasedRunner', max_epochs=273)
auto_scale_lr = dict(base_batch_size=64)