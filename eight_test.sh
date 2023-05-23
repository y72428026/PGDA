train_dataset=T
epoch=146
tag=Focal
resolution=640

for test_dataset in T L
do
    config_dir=/home/yebh/mmdet2/work_dirs/baseline/yolov3-${tag}-${resolution}-${test_dataset}-3class-000-0-4gpu-dist/yolov3-${tag}-${resolution}-${test_dataset}-3class-000-0.py
    model_dir=/home/yebh/mmdet2/work_dirs/baseline/yolov3-${tag}-${resolution}-${train_dataset}-3class-000-0-4gpu-dist/epoch_${epoch}.pth
    log_dir=/home/yebh/mmdet2/work_dirs/baseline/yolov3-${tag}-${resolution}-${train_dataset}-3class-000-0-4gpu-dist/0_0th_${epoch}_${test_dataset}_APARF.log
    out_dir=/home/yebh/mmdet2/work_dirs/baseline/yolov3-${tag}-${resolution}-${train_dataset}-3class-000-0-4gpu-dist/0_0th_${epoch}_${test_dataset}.pkl
    gpu_ids=7
    CUDA_VISIBLE_DEVICES=${gpu_ids} python tools/test.py \
        ${config_dir} \
        ${model_dir} \
        --eval bbox \
        --eval-options "classwise=True" \
        --log_dir ${log_dir}\
        --out  ${out_dir}
    python tools/analysis_tools/confusion_matrix.py \
        ${config_dir} \
        ${out_dir}\
        /home/yebh/mmdet2/work_dirs/baseline/yolov3-${tag}-${resolution}-${train_dataset}-3class-000-0-4gpu-dist/ \
        --tag 0_0th_${epoch}_${test_dataset}
done
