_base_ = [  '../yolo/uda_yolov3_608_L2T_SCL-000-0.py', ]
fp16 = dict(loss_scale='dynamic')
samples_per_gpu=32

model = dict(
    type='YOLOV3',
    backbone=dict(
        out_indices=(3, 4, 5)),
    bbox_head=dict(
        wscl=False)
)

uda=dict(
    da_head=dict(
        img_weight=100
    ),
    da_ano_head=dict(
        img_weight=100
    ),
    da_pred_head=dict(
        img_weight=0
    ),
    cfa_thres=0.5,
    enable_category_loss=False,
    category_weight=0,
    )

data = dict(
    samples_per_gpu=samples_per_gpu)
