_base_ = [  '../../../yolo/uda_yolov3_608_L2T_SCL_Xohem-100e-000-0-fp16.py', ]


uda=dict(
    da_head=dict(
        img_weight=100,
    ),
    da_ano_head=dict(
        img_weight=0,
    ),
    da_pred_head=dict(
        img_weight=0,
    ),
    cfa_thres=0.5,
    enable_category_loss=True,
    category_weight=0.0875,
    alpha_mb=1,
    cfa_v=14,
    )

