# HPT2HPL
dataset=HPT2HPL
work_dir=/data/yebh/mmdet2/work_dirs/BIS/${dataset}
config_dir='/data/yebh/mmdet2/work_dirs/BIS/HPT2HPL/CFA_1k1k1k0/yolov3-UDA-640-HPT2HPL-3class-1k1k1k0-cfav9-0.025-cT0.5-pT0.5-a0.1-v1/yolov3-UDA-640-HPT2HPL-3class-1k1k1k0-cfav9-0.025-cT0.5-pT0.5-a0.1.py'
model_dir=/data/yebh/mmdet2/work_dirs/BIS/HPT2HPL/yolov3-UDA-640-HPT2HPL-3class-DA-462-462-462-0-cfav9-0.075-cT0.6-pT0.7-a0.6-v1_epoch_81.pth

# # xqpink2xqblue
# dataset=XQpink2XQblue
# work_dir=/data/yebh/mmdet2/work_dirs/BIS/${dataset}
# config_dir='/data/yebh/mmdet2/work_dirs/BIS/XQpink2XQblue/CFA_cT/yolov3-UDA-640-XQpink2XQblue-5class-DA-2150-2150-2150-0-cfav9-0.0075-cT0.6-pT0.6-a0.1-v1/yolov3-UDA-640-XQpink2XQblue-5class-DA-2150-2150-2150-0-cfav9-0.0075-cT0.6-pT0.6-a0.1.py'
# model_dir='/data/yebh/mmdet2/work_dirs/BIS/XQpink2XQblue/yolov3-UDA-640-XQpink2XQblue-5class-DA-2150-2150-2150-0-cfav9-0.0075-cT0.9-pT0.6-a0.1-v1_epoch_69.pth'

# # XQXY2XQY
# dataset=XQXY2XQY
# work_dir=/data/yebh/mmdet2/work_dirs/BIS/${dataset}
# config_dir='/data/yebh/mmdet2/work_dirs/BIS/XQXY2XQY/DA_aba/yolov3-UDA-640-XQXY2XQY-5class-DA-462-462-462-0-cfav9-0.01-cT0.7-pT0.7-a0.1-v3/yolov3-UDA-640-XQXY2XQY-5class-DA-462-462-462-0-cfav9-0.01-cT0.7-pT0.7-a0.1.py'
# model_dir='/data/yebh/mmdet2/work_dirs/BIS/XQXY2XQY/yolov3-UDA-640-XQXY2XQY-5class-DA-462-462-462-0-cfav9-0.0125-cT0.7-pT0.7-a0.1-v2-epoch_68.pth'

log_dir=${work_dir}/${epoch}_show_img.log
show_dir=${work_dir}/../my_result/${dataset}
gpu_ids=0
### test class-wise **AP50** and ***save*** the prediction images
echo python tools/test_tsne.py \
    ${config_dir} \
    ${model_dir} \
    --gpu-id ${gpu_ids} \
    --eval bbox \
    --eval-options "classwise=True" "iou_thrs=[0.5]" \
    --log_dir ${log_dir} \
    --show-dir ${show_dir}  
    # --font_size 30 --thickness 10 

# 可视化GT需要做以下事情：
# 1. test_pipeline中加入dict(type='LoadAnnotations', with_bbox=True),
# 2. code/mmdetection/mmdet/datasets/custom.py-》prepare_test_img中加入anno_info,和1配合读取bbox信息；
# 3. 将MultiScaleFlipAug的img_scale改成img_scale=(640, 640)（原来图像的大小）；
# 4. code/mmdetection/mmdet/apis/test.py-》single_gpu_test：改batchsize的设置，屏蔽model（data），屏蔽model.module.show_result，放出model.module.show_gt_result
# bbox_palette = palette_val([(255, 87, 51),(255, 189, 51),(219, 255, 51),(117, 255, 51),(123, 128, 235),(240, 139, 240)])
# 可选
# 1. code/mmdetection/mmdet/core/visualization/image.py-》imshow_det_bboxes，调整classname，bbox_palette，text_palette；
# 2. code/mmdetection/mmdet/models/detectors/base.py-》show_result：调整thickness和font_size



# config="yolox_l_8x8_300e_T2L" 
# model="yolox"
# model_echo="epoch_150.pth"
# gpu_ids=7


# ### define sevaral path and name
# work_dir="work_dirs/${config}/"
# test_dir="test_dirs/${work_dir}"
# show_dir="${test_dir}/${config}/images/"
# config_name="${config}.py"
# log_name="${config}.log"
# # profile_name="${config}_profile.log"
# log_dir=${test_dir}${log_name}
# # prifile_dir=${test_dir}${profile_name}
# config_dir=${work_dir}${config_name}
# model_dir=${work_dir}${model_echo}

# # ### test COCO AP benchmarks
# # # python -m cProfile -o ${profile_dir} tools/test.py \
# # python tools/test.py \
# #     ${config_dir} \
# #     ${model_dir} \
# #     --gpu-ids ${gpu_ids} \
# #     --eval bbox \
# #     --log_dir ${log_dir} 

# ### test class-wise **AP50** and ***save*** the prediction images
# python tools/test.py \
#     ${config_dir} \
#     ${model_dir} \
#     --gpu-ids ${gpu_ids} \
#     --eval bbox \
#     --log_dir ${log_dir} \
#     --eval-options "classwise=True" "iou_thrs=[0.5]" \
#     --show-dir ${show_dir} \
#     --font_size 30 --thickness 10 \
    

# # ### test class-wise ***6 AP*** metrics and ***save*** the prediction images
# # python tools/test.py \
# #     work_dirs/ssd300_seaship/ssd300_seaship.py \
# #     work_dirs/ssd300_seaship/latest.pth \
# #     --gpu-ids=7 \
# #     --eval bbox \
# #     --eval-options "classwise=True" "logger=test_dirs/ssd300_seaship.log" \
# #     --show-dir test_dirs/ssd300_seaship \
    