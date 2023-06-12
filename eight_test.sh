
train_dataset=PCBCropGray
# list all the test dataset
test_datasets=(DeepPCB) #PCBCropGray)

# define the epoch and resolution
resolution=640
root_dir=/data/yebh
mmdet2_dir=${root_dir}/mmdet2
# load the first line of txt file to get $epoch and define model_dir
txt_dir=${mmdet2_dir}/work_dirs/PCB/baseline/yolov3-${resolution}-${train_dataset}*/*.txt
line=$(head -n 1 ${txt_dir})
# split the line by space
IFS=',' read -r -a array <<< "$line"
# get the first element of array
epoch=${array[0]:1}
# define model_dir and config_dir
model_dir=${mmdet2_dir}/work_dirs/PCB/baseline/yolov3-${resolution}-${train_dataset}*/epoch_${epoch}.pth
config_dir=${mmdet2_dir}/work_dirs/PCB/baseline/yolov3-${resolution}-${train_dataset}*/yolov3-${resolution}-${train_dataset}.py

for test_dataset in ${test_datasets[@]}
# for test_dataset in DeepPCB PCBCropGray
do
    # echo ${test_dataset}
    log_dir=${mmdet2_dir}/work_dirs/PCB/baseline/yolov3-${resolution}-${train_dataset}*/_${epoch}_${test_dataset}_APARF.log
    out_dir=${mmdet2_dir}/work_dirs/PCB/baseline/yolov3-${resolution}-${train_dataset}*/_${epoch}_${test_dataset}.pkl
    gpu_ids=0

    CUDA_VISIBLE_DEVICES=${gpu_ids} python ${mmdet2_dir}/tools/test.py \
        ${config_dir} \
        ${model_dir} \
        --eval bbox \
        --eval-options "classwise=True" \
        --log_dir ${log_dir}\
        --out  ${out_dir} \
        --cfg-options \
        data.test.img_prefix=${root_dir}/dataset/${test_dataset}/JPEGImages \
        data.test.ann_file=${root_dir}/dataset/${test_dataset}/${test_dataset}_test.json

    python ${mmdet2_dir}/tools/analysis_tools/confusion_matrix.py \
        ${config_dir} \
        ${out_dir}\
        ${mmdet2_dir}/work_dirs/PCB/baseline/yolov3-${resolution}-${train_dataset}*/ \
        --tag _${epoch}_${test_dataset} \
        --cfg-options \
        data.test.img_prefix=${root_dir}/dataset/${test_dataset}/JPEGImages \
        data.test.ann_file=${root_dir}/dataset/${test_dataset}/${test_dataset}_test.json
done
