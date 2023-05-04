_base_ = [  '../../yolo/uda_yolov3_608_L2T_SCL-100e-000-0.py', ]


uda=dict(
    da_head=dict(
        img_weight=1 
    ),
    da_ano_head=dict(
        img_weight=1
    ),
    da_pred_head=dict(
        img_weight=0
    ),
    cfa_thres=0.5,
    enable_category_loss=False,
    category_weight=0,
    alpha_mb=1,
    )

