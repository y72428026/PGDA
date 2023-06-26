_base_ = [  '../../yolo/yolov3-UDA-640-XQpink2XQblue-5class-0000-0.py']

uda = dict(
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
    cfa_conf_thres=0.5,
    cfa_pred_thres=0.5,
    enable_category_loss=True,
    category_weight=0.01,
    alpha_mb=0.1,
    cfa_v=9,
)
