_base_ = [  '../yolo/uda_yolov3_T2L_SCL-000-0.py',
            ]

uda=dict(
    da_head=dict(
        img_weight=7500,
        GAN_type='WGAN'  
    ),
    da_ano_head=dict(
        img_weight=0,
        GAN_type='WGAN'  
    ),
    da_pred_head=dict(
        img_weight=0,
        GAN_type='WGAN'  
    ),
    enable_category_loss=False,
    category_weight=0.0625,
    )

