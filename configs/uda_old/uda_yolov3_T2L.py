_base_ = [  '../yolo/yolov3_d53_mstrain-608_273e_coco.py',
            # '../_base_/default_runtime.py',
            # '../_base_/schedules/adamw.py',
            # '../_base_/schedules/poly10warm.py'
            ]
model = dict(
    bbox_head=dict(
        num_classes=6,
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(439, 110), (220, 243), (419, 153)],
                        [(135, 104), (144, 189), (334, 94)],
                        [(31, 28), (33, 123), (70, 61)]],
            strides=[32, 16, 8])
    )   
)
uda=dict(
    type='UDAModel',
    da_head=dict(
        type='DomainAdaptationHead',
        in_channels=[512, 256, 128],
        img_weight=1000
    ),
    da_ano_head=dict(
        type='DomainAdaptationHead',
        in_channels=[1024, 512, 256],  
        img_weight=1000
    ),
    da_pred_head=dict(
        type='DomainAdaptationHead',
        in_channels=[33, 33, 33],  
        img_weight=0
    ),
    enable_category_loss=False,
    category_weight=0.1,
    enable_ease_loss=False,
    ease_weight=1e-5,
    auxiliary_head_num=0,
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
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
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
# dataset_type = 'CocoDataset'
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
        _delete_=True,
        type='UDADataset',
        source=dict(
            type=dataset_type,
            classes=classes,
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_all.json',
            pipeline=train_pipeline),
        target=dict(
            type=dataset_type,
            classes=classes,
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_trainval.json',
            # ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json',
            pipeline=train_pipeline),
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_test.json',
        # ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
        ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_test.json',
        # ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json',
        pipeline=test_pipeline))

checkpoint_config = dict(interval=1)
cudnn_benchmark = True

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
# lr
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[218, 246])

runner = dict(type='EpochBasedRunner', max_epochs=273)
auto_scale_lr = dict(base_batch_size=64)