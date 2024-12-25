_base_ = [  './yolov3_d53_mstrain-608_273e_coco.py', ]

# fp16 = dict(loss_scale='dynamic')
# samples_per_gpu=32
samples_per_gpu=8
workers_per_gpu=4
evaluation = dict(interval=1, metric=['bbox'])
log_config = dict(interval=30)
find_unused_parameters=True

image_scale=(640,640)
dataset_type='PCB'
mode = 'uda_640'
num_classes=6
source_dataset = 'PCBCropGray'
target_dataset = 'DeepPCB'
classes = ('open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole')

import os
trainpath = os.getcwd()
root_dir = trainpath[:trainpath.find('yebh')]+'yebh/'
load_from = root_dir + f'/checkpoint/{source_dataset}/best_addmodel.pth'
model = dict(
    type='YOLOV3',
    bbox_head=dict(
        type='YOLOV3Head_mine',
        num_classes=num_classes,
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
    # da_backbone_head=dict(
    #     type='DomainAdaptationHead',
    #     in_channels=[1024, 512, 256],
    #     GAN_type='LSGAN',
    #     img_weight=0,  
    # ),
    da_head=dict(
        type='DomainAdaptationHead',
        in_channels=[512, 256, 128],
        GAN_type='LSGAN',
        img_weight=0,  
    ),
    da_ano_head=dict(
        type='DomainAdaptationHead',
        in_channels=[1024, 512, 256],  
        GAN_type='LSGAN',
        # in_channels=[1024+128, 512+128, 256+128],  
        img_weight=0
    ),
    da_pred_head=dict(
        type='DomainAdaptationHead',
        in_channels=[33, 33, 33],  
        img_weight=0
    ),
    cfa_thres=0.5,
    enable_category_loss=False,
    category_weight=0,
    enable_ease_loss=False,
    ease_weight=1e-5,
    auxiliary_head_num=0,
    num_classes=num_classes,
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
    dict(type='Resize', img_scale=[(320, 320), image_scale], keep_ratio=True),
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
        img_scale=image_scale,
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

dataset_type = 'CocoDataset'


data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict( 
        _delete_=True,
        type='UDADataset',
        source=dict(
            type=dataset_type,
            classes=classes,
            img_prefix=root_dir+f'dataset/{source_dataset}/JPEGImages',
            ann_file=root_dir+f'dataset/{source_dataset}/{source_dataset}_all.json',
            pipeline=train_pipeline),
        target=dict(
            type=dataset_type,
            classes=classes,
            img_prefix=root_dir+f'dataset/{target_dataset}/JPEGImages',
            ann_file=root_dir+f'dataset/{target_dataset}/{target_dataset}_trainval.json',
            pipeline=train_pipeline),
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=root_dir+f'dataset/{target_dataset}/JPEGImages',
        ann_file=root_dir+f'dataset/{target_dataset}/{target_dataset}_test.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix=root_dir+f'dataset/{target_dataset}/JPEGImages',
        ann_file=root_dir+f'dataset/{target_dataset}/{target_dataset}_test.json',
        pipeline=test_pipeline))


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