import subprocess
import pynvml
import random
import sys

# Set the "-e" option
subprocess.run("set -e", shell=True)


def choose_available_gpu() -> int:
    # judge the status of GPU : if GPU is available,
    # then scan the queue( can by folder json etc.) and run the job, return the result to user
    percent_not_used = 0.02  # 10% of GPU memory is not used
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    # print("GPU count: %d" % count)
    for i in range(count):
        # print("GPU %d" % i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print("GPU used memory: %d" % info.used)
        # print("GPU all memory: %d" % info.total)
        # print("GPU free memory: %d" % info.free)
        percent_used = float(info.used) / float(info.total) * 100
        print('GPU %d used memory percent: %f' % (i, percent_used))
        if float(info.used) / float(info.total) < percent_not_used:
            pynvml.nvmlShutdown()
            print("GPU %i is available" % i)
            # input("Press Enter to continue...")
            return i
    pynvml.nvmlShutdown()
    print("GPU is not available")
    time.sleep(1800)
    return None


# Set variables
GPUS_num = 2
gpu = choose_available_gpu()
# gpu=6
# GPUS = "0,1" GPUS = "2,3"
PORT = random.randint(29500, 29599)
source_dataset = "XQpink"
target_dataset = "XQblue"
dataset = f"{source_dataset}2{target_dataset}"
resolution = 640

# root_dir = "/data/yebh/mmdet2"
root_dir = sys.path[0].split("mmdet2")[0] + 'mmdet2'
dataset_type = "BIS"
version = 1
tag = ''

cfav = 9
conf_T = 0
pred_T = 0
a = 0.1
cfa_weight = 0
alist = [100, 215, 462, 1000, 2150, 4620, 10000]
DA_list = [f'{ele}-{ele}-{ele}-0' for ele in alist]
DA = DA_list[gpu]
model_tag = "UDA"
tag = '5class'
dataset_tag = f'XQ{tag}'
cfg_tag = f'{tag}'
fp16 = ''
for version in [1, 2, 3]:
    config_dir = f"{root_dir}/configs/{dataset_type}/{dataset}/yolov3-{model_tag}-{resolution}-{dataset}-{cfg_tag}-DA-{DA}-cfav{cfav}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}.py"
    work_dir = f"{root_dir}/work_dirs/{dataset_type}/{dataset}/yolov3-{model_tag}-{resolution}-{dataset}-{cfg_tag}-DA-{DA}-cfav{cfav}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}"
    work_dir = f"{work_dir}-v{version}"

    subprocess.run(f"python {root_dir}/tools/train.py "
                f"{config_dir} --work-dir={work_dir} --gpu-id={gpu} --auto-scale-lr --seed=1079546523", shell=True)

    # subprocess.run(f"CUDA_VISIBLE_DEVICES={GPUS} python -m torch.distributed.launch "
    #                f"--nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node={GPUS_num} "
    #                f"--master_port={PORT} {root_dir}/tools/train.py "
    #                f"{config_dir} --work-dir={work_dir} --auto-scale-lr --seed=1079546523 "
    #                f"--launcher pytorch", shell=True)
    subprocess.run(
        f"python {root_dir}/read_json_and_save_topk.py --path={work_dir} --gpu={gpu}", shell=True)
    subprocess.run(f"python {sys.path[0]}/test.py --source_dataset={source_dataset} --target_dataset {target_dataset} "
                   f"--config_dir={config_dir} --dataset_tag={dataset_tag} --work_dir={work_dir} --gpu={gpu}", shell=True)
