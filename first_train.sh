GPUS_num=2
# GPUS=1,2,3,4
GPUS=4,5
# GPUS=2,3
# GPUS=4,5,6,7
PORT=29503
gpu=1

tag=SCL-Focal
location=baseline

for dataset in T
# for dataset in L2T T2L 
do
    # location=dataset
    configs=configs/uda_640/${location}/yolov3-${tag}-640-${dataset}-3class-*.py
    work_dir=work_dirs/${location}/yolov3-${tag}-640-${dataset}-3class
    work_dir=${work_dir}-4gpu-dist

    python tools/train.py \
        $configs --work-dir=${work_dir} --gpu-id=${gpu} --auto-scale-lr --seed=1079546523 --testcfg\

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
        --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS_num} \
        --master_port=${PORT} ./tools/train.py \
        $configs --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 \
        --launcher pytorch ${@:3}
    python read_json_and_save_topk.py --path=${work_dir}
done

# GPUS=4,5,6,7
# PORT=29504
# for dataset in L
# do
#     configs=configs/uda_640/baseline/yolov3-640-${dataset}-000-0.py
#     work_dir=work_dirs/baseline/yolov3-640-${dataset}-000-0
#     work_dir=${work_dir}-4gpu-dist

#     PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#     CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
#         --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS_num} \
#         --master_port=${PORT} ./tools/train.py \
#         $configs --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 \
#         --launcher pytorch ${@:3}

#     python read_json_and_save_topk.py --path=${work_dir}
# done