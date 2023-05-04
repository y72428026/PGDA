GPUS_num=4
# GPUS=0,1,2,3
GPUS=4,5,6,7
PORT=29503
configs=/home/yebh/mmdetection/configs/uda_608/uda_yolov3_T2L_SCL_75k75k0-00625-a07.py
work_dir=work_dirs/test_dataset/T2L_SCL/CFA_608/uda_yolov3_T2L_SCL-75k75k0-00625-a07-updatelater-v11-v0
work_dir=${work_dir}-4gpu-bs16-fp32

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS_num} \
    --master_port=${PORT} ./tools/train.py \
    $configs --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 \
    --launcher pytorch ${@:3}

python read_json_and_save_top50.py --path=${work_dir}

