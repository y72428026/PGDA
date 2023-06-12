_base_ = [  '../../yolo/yolov3-UDA-640-PCBCropGray2DeepPCB-000-0.py']


uda=dict(
    da_head=dict(
        img_weight=7500,
    ),
    da_ano_head=dict(
        img_weight=7500,
    ),
    da_pred_head=dict(
        img_weight=0,
    ),
    cfa_conf_thres=0.3,
    cfa_pred_thres=0.7,
    enable_category_loss=True,
    category_weight=0.0625,
    alpha_mb=0.1,
    cfa_v=9,
    )

