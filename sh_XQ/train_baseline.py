import subprocess, pynvml, time, os, sys
import random

# Set the "-e" option
subprocess.run("set -e", shell=True)

def choose_available_gpu() -> int:
    # judge the status of GPU : if GPU is available, 
    # then scan the queue( can by folder json etc.) and run the job, return the result to user
    percent_not_used = 0.1 # 10% of GPU memory is not used
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
GPUS = "0,1,2,3"
# GPUS = "4,5,6,7"
GPUS_num = 4
PORT = random.randint(29500, 29599)

resolution = 640
root_dir = "/data/yebh/mmdet2"
dataset_type = "BIS"
source_dataset = "XQXY"
for tag in ['7class', '5class']:
    dataset_tag = f'XQ{tag}'
    cfg_tag = f'-{tag}'
    
    # get target dataset and dataset
    if source_dataset in ["XQpink", "XQpurple", "XQblue"]:
        target_dataset = "XQpurple XQpink XQblue"
    if source_dataset in ["XQY", "XQXY"]:
        target_dataset = "XQY XQXY"
    dataset = source_dataset

    config_dir = f"{root_dir}/configs/{dataset_type}/{dataset}/yolov3-{resolution}-{dataset}{cfg_tag}.py"
    work_dir = f"{root_dir}/work_dirs/{dataset_type}/{dataset}/yolov3-{resolution}-{dataset}{cfg_tag}"
    
    # subprocess.run(f"python {root_dir}/tools/train.py "
    #             f"{config_dir} --work-dir={work_dir} --gpu-id={gpu} --auto-scale-lr --seed=1079546523", shell=True)

    subprocess.run(f"CUDA_VISIBLE_DEVICES={GPUS} python -m torch.distributed.launch "
                    f"--nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node={GPUS_num} "
                    f"--master_port={PORT} {root_dir}/tools/train.py "
                    f"{config_dir} --work-dir={work_dir} --auto-scale-lr --seed=1079546523 "
                    f"--launcher pytorch", shell=True)

    gpu = choose_available_gpu()
    subprocess.run(f"python {root_dir}/read_json_and_save_topk.py --path={work_dir} --gpu={gpu}", shell=True)
    subprocess.run(f"python {sys.path[0]}/test.py --source_dataset={source_dataset} --target_dataset {target_dataset} "
                f"--config_dir={config_dir} --dataset_tag={dataset_tag} --work_dir={work_dir} --gpu={gpu}", shell=True)



