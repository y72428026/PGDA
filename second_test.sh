work_dir=/home/yebh/work_dirs_remote/test_dataset/L2T_SCL_608/ablation-a-T05/uda_yolov3_L2T_SCL_1h1h0-00875-T05-a05-gpu7-v0-247-247-246
# config_name="uda_yolov3_T2L_SCL-75k75k0-00625_test_source_dataset.py"
config_name=uda_yolov3_L2T_SCL_1h1h0-00875-T05-a05.py
epoch=235
  
config_dir=${work_dir}/${config_name}
model_dir=${work_dir}/epoch_${epoch}.pth
# log_dir=${work_dir}/${epoch}_test_608_target_category-wise-APAR.log
log_dir=${work_dir}/${epoch}_test_608_target_category-wise-APAR_mine.log
show_dir=${work_dir}/image_${epoch}_target
gpu_ids=7
### test class-wise **AP50** and ***save*** the prediction images
CUDA_VISIBLE_DEVICES=${gpu_ids} python tools/test.py \
    ${config_dir} \
    ${model_dir} \
    --eval bbox \
    --log_dir ${log_dir} \
    --eval-options "classwise=True" 
    # --gpu-id ${gpu_ids} 
    # "iou_thrs=[0.5]" 
    # --show-dir ${show_dir}  
    # --font_size 30 --thickness 10 

# 可视化GT需要做以下事情：
# 1. test_pipeline中加入dict(type='LoadAnnotations', with_bbox=True),
# 2. code/mmdetection/mmdet/datasets/custom.py-》prepare_test_img中加入anno_info,和1配合读取bbox信息；
# 3. 将MultiScaleFlipAug的img_scale改成img_scale=(640, 640)（原来图像的大小）；
# 4. code/mmdetection/mmdet/apis/test.py-》single_gpu_test：改batchsize的设置，屏蔽model（data），屏蔽model.module.show_result，放出model.module.show_gt_result
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
    