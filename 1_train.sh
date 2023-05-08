gpu=1

cfg=13
T=05

# for a in 01 1
# do
#     for version in 1 2 3
#     do
#         configs=./configs/uda_608/100e/uda_yolov3_LHE2T_SCL-100e-1h1h0-00875-T${T}-a${a}-cfa${cfg}.py
#         work_dir=./work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_LHE2T_SCL-100e-00875-T${T}-a${a}-cfav${cfg}-ohem
#         work_dir=${work_dir}-gpu${gpu}-v$version
#         python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 --cfg-options \
#             data.samples_per_gpu=8
#         python read_json_and_save_top50.py --path=${work_dir} --gpu=${gpu}
#     done
# done

GPUS_num=4
GPUS=1,5,6,7
PORT=29503


for a in 01 1
do
    for version in 1 2 3
    do
        configs=./configs/uda_608/100e/uda_yolov3_LHE2T_SCL-100e-1h1h0-00875-T${T}-a${a}-cfa${cfg}.py
        work_dir=./work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_LHE2T_SCL-100e-00875-T${T}-a${a}-cfav${cfg}-ohem
        work_dir=${work_dir}-dist-v$version
        CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch \
            --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS_num} \
            --master_port=${PORT} ./tools/train.py \
            $configs --work-dir=${work_dir} --auto-scale-lr --seed=1079546523  \
            --launcher pytorch ${@:3} --cfg-options \
            data.samples_per_gpu=8
    done
done