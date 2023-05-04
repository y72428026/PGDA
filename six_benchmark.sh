gpu='0'
# config='work_dirs/test_dataset/L2T_SCL/cate-1h1h/yolo-da-L2T-newanchor-da-mse-1h-1h-0-v00875-sclv2-64-cn4-v5-254-250/uda_yolov3_L2T_SCL-1h1h0-00875_for_test.py'
# checkpoint='work_dirs/test_dataset/L2T_SCL/cate-1h1h/yolo-da-L2T-newanchor-da-mse-1h-1h-0-v00875-sclv2-64-cn4-v5-254-250/epoch_113.pth'
# checkpoint='/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/faster-baseline/epoch_11.pth'
config='/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/a+b/yolo-da-L2T-newanchor-baseline-Xsclv2-v0-cn3-241/uda_yolov3_L2T-baseline_640.py'
checkpoint='/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/a+b/yolo-da-L2T-newanchor-baseline-Xsclv2-v0-cn3-241/epoch_86.pth'
CUDA_VISIBLE_DEVICE=$gpu python -m torch.distributed.launch tools/analysis_tools/benchmark.py $config $checkpoint \
--launcher pytorch --repeat-num 10 --fuse-conv-bn

# CONFIG=$config
# CHECKPOINT=$checkpoint
# GPUS=$gpu
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/analysis_tools/benchmark.py \
#     $CONFIG \
#     $CHECKPOINT \
#     --launcher pytorch \
#     ${@:4}