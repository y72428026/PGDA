# # HPT2HPL
# dataset=HPL
# work_dir=/data/yebh/mmdet2/work_dirs/BIS/${dataset}
# config_dir=/data/yebh/mmdet2/configs/BIS/HPL/yolov3-640-HPL-3class-gt.py


# xqpink2xqblue
dataset=XQblue
work_dir=/data/yebh/mmdet2/work_dirs/BIS/${dataset}
config_dir=/data/yebh/mmdet2/configs/BIS/XQblue/yolov3-640-XQblue-4class-gt.py


# # XQXY2XQY
# dataset=XQY
# work_dir=/data/yebh/mmdet2/work_dirs/BIS/${dataset}
# config_dir=/data/yebh/mmdet2/configs/BIS/XQY/yolov3-640-XQY-5class-gt.py

log_dir=${work_dir}/${epoch}_show_img.log
show_dir=${work_dir}/../gt_result/${dataset}
gpu_ids=7
### test class-wise **AP50** and ***save*** the prediction images
python tools/test_gt.py \
    ${config_dir} \
    ${model_dir} \
    --gpu-id ${gpu_ids} \
    --eval bbox \
    --eval-options "classwise=True" "iou_thrs=[0.5]" \
    --log_dir ${log_dir} \
    --show-dir ${show_dir}  \
    # --font_size 30 --thickness 10 

# 可视化GT需要做以下事情：
# 1. test_pipeline中加入dict(type='LoadAnnotations', with_bbox=True),
# 2. code/mmdetection/mmdet/datasets/custom.py-》prepare_test_img中加入anno_info,和1配合读取bbox信息；
# 3. 将MultiScaleFlipAug的img_scale改成img_scale=(640, 640)（原来图像的大小）；
# 4. code/mmdetection/mmdet/apis/test.py-》single_gpu_test：改batchsize的设置，屏蔽model（data），屏蔽model.module.show_result，放出model.module.show_gt_result
# 可选
# 1. code/mmdetection/mmdet/core/visualization/image.py-》imshow_det_bboxes，调整classname，bbox_palette，text_palette；
# 2. code/mmdetection/mmdet/models/detectors/base.py-》show_result：调整thickness和font_size
