set -e
GPUS_num=4
gpu=5

GPUS=0,1,2,3
# GPUS=4,5,6,7
PORT=$(($RANDOM % 100 + 29500))

for dataset in PCBCropGray # DeepPCB # PCBCrop
do
    configs=configs/PCB/baseline/yolov3-640-${dataset}${amp}.py
    work_dir=work_dirs/PCB/baseline/yolov3-640-${dataset}${amp}-test

    # python tools/train.py \
    #     $configs --work-dir=${work_dir} --gpu-id=${gpu} --auto-scale-lr --seed=1079546523 --testcfg\

    # python tools/train.py \
    #     $configs --work-dir=${work_dir} --gpu-id=${gpu} --auto-scale-lr --seed=1079546523 \

    work_dir=${work_dir}-${GPUS_num}gpu-dist
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
        --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS_num} \
        --master_port=${PORT} ./tools/train.py \
        $configs --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 \
        --launcher pytorch ${@:3}

    python read_json_and_save_topk.py --path=${work_dir} --gpu=${GPUS:0:1}
done
