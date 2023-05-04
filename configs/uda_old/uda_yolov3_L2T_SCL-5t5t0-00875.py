_base_ = [  '../yolo/yolov3_d53_mstrain-608_273e_coco.py',
            # '../_base_/default_runtime.py',
            # '../_base_/schedules/adamw.py',
            # '../_base_/schedules/poly10warm.py'
            ]

model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(1, 2, 3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck_mine',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head_mine',
        num_classes=6,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
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
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))

uda=dict(
    type='UDAModel_SCL',
    da_head=dict(
        type='DomainAdaptationHead',
        in_channels=[512, 256, 128],
        img_weight=50
    ),
    da_ano_head=dict(
        type='DomainAdaptationHead',
        in_channels=[1024, 512, 256],  
        # in_channels=[1024+128, 512+128, 256+128],  
        img_weight=50
    ),
    da_pred_head=dict(
        type='DomainAdaptationHead',
        in_channels=[33, 33, 33],  
        img_weight=0
    ),
    enable_category_loss=True,
    category_weight=0.0875,
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
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/JPEGImages',
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_13_16/HP_13_16_all.json',
            pipeline=train_pipeline),
        target=dict(
            type=dataset_type,
            classes=classes,
            img_prefix='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/JPEGImages',
            ann_file='/media/data3/yebh/dataset/BIS/BIS_HP_transparent/HP_transparent_trainval.json',
            pipeline=train_pipeline),
        ),
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

# checkpoint_config = dict(interval=10)
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