_base_ = [  '../../yolo/uda_yolov3_608_T2L_SCL-000-0.py', ]


uda=dict(
    da_head=dict(
        img_weight=25  
    ),
    da_ano_head=dict(
        img_weight=25
    ),
    da_pred_head=dict(
        img_weight=0
    ),
    cfa_thres=0.3,
    enable_category_loss=True,
    category_weight=0.0625,
    alpha_mb=0.7
    )

