_base_ = [  '../../yolo/yolov3-UDA-SCL-640-L2T-3class-000-0.py', ]


uda=dict(
    da_head=dict(
        img_weight=0  
    ),
    da_ano_head=dict(
        img_weight=0
    ),
    da_pred_head=dict(
        img_weight=0
    ),
    cfa_conf_thres=0.5,
    cfa_pred_thres=0.5,
    enable_category_loss=False,
    category_weight=0,
    alpha_mb=1,
    cfa_v=9
    )

