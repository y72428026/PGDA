set -e
GPUS_num=4
gpu=5

GPUS=0,1,2,3
# GPUS=4,5,6,7

PORT=$(($RANDOM % 20 + 29500))
source_dataset=RSDDII
target_dataset=RSDDI
dataset=${source_dataset}2${target_dataset}
model_tag=UDA
resolution=200

cfav=9
conf_T=0.3
pred_T=0.7
a=0.1
DA=75k75k0
cfa_weight=0.0625
root_dir=/data/yebh/mmdet2
dataset_type=RSDD
for version in 1 #2 3
do
    config_dir=${root_dir}/configs/${dataset_type}/${dataset}/yolov3-${model_tag}-${resolution}-${dataset}-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}.py
    work_dir=${root_dir}/work_dirs/${dataset_type}/${dataset}/yolov3-${model_tag}-${resolution}-${dataset}-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}

    # work_dir=${work_dir}-gpu${gpu}-v$version
    # python ../tools/train.py \
    #     $config_dir --work-dir=${work_dir} --gpu-id=${gpu} --auto-scale-lr --seed=1079546523 --testcfg\

    work_dir=${work_dir}-gpu${gpu}-v$version
    # python ../tools/train.py \
    #     $config_dir --work-dir=${work_dir} --gpu-id=${gpu} --auto-scale-lr --seed=1079546523 \

    # work_dir=${work_dir}-${GPUS_num}gpu-dist-v$version
    # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    # CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
    #     --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS_num} \
    #     --master_port=${PORT} ../tools/train.py \
    #     $config_dir --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 \
    #     --launcher pytorch ${@:3}

    python ${root_dir}/read_json_and_save_topk.py --path=${work_dir} --gpu=${GPUS:0:1}
    python test.py --source_dataset=${source_dataset} --target_dataset=${target_dataset} \
    --config_dir=${config_dir} \
    --work_dir=${work_dir} --gpu=${GPUS:0:1} 
    # --model_tag=${model_tag} \
    # --version=${version} \
    # --resolution=${resolution} --DA=${DA} --cfav=${cfav} --cfa_weight=${cfa_weight} \
    # --conf_T=${conf_T} --pred_T=${pred_T} --a=${a} 

done
