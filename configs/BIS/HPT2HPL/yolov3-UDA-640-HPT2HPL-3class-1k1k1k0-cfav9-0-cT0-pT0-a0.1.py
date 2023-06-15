_base_ = [  '../../yolo/yolov3-UDA-640-HPT2HPL-3class-0000-0.py']

uda=dict(
    da_backbone_head=dict(
        img_weight=1000,
    ),
    da_neck_head=dict(
        img_weight=1000,
    ),
    da_pred_head=dict(
        img_weight=1000,
    ),
    da_output_head=dict(
        img_weight=0,
    ),
    cfa_conf_thres=0,
    cfa_pred_thres=0,
    enable_category_loss=False,
    category_weight=0,
    alpha_mb=0.1,
    cfa_v=9,
    )

