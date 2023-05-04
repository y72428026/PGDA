_base_ = [  '../yolo/uda_yolov3_T2L_SCL-000-0.py',
            ]

uda=dict(
    da_head=dict(
        img_weight=7500  
    ),
    da_ano_head=dict(
        img_weight=7500
    ),
    da_pred_head=dict(
        img_weight=0
    ),
    cfa_thres=0.9,
    enable_category_loss=True,
    category_weight=0.0625,
    )

