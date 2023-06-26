_base_ = [  '../../yolo/yolov3-UDA-640-HPT2HPL-3class-0000-0.py']

uda=dict(
    da_backbone_head=dict(
        img_weight=462,
    ),
    da_neck_head=dict(
        img_weight=462,
    ),
    da_pred_head=dict(
        img_weight=462,
    ),
    da_output_head=dict(
        img_weight=0,
    ),
    cfa_conf_thres=0.1,
    cfa_pred_thres=0.7,
    enable_category_loss=True,
    category_weight=0.075,
    alpha_mb=0.2,
    cfa_v=9,
    )

