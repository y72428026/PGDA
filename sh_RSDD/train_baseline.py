import subprocess
import random

# Set the "-e" option
subprocess.run("set -e", shell=True)

# Set variables
GPUS_num = 4
gpu = 0
GPUS = "0,1,2,3"
PORT = random.randint(29500, 29599)
# dataset = "RSDDIIO"
dataset = "NEU3A"
source_dataset = dataset
target_dataset = "RSDDI RSDDII NEUA RSDDIO RSDDIIO RSDDIO RSDDIIO"
# print(target_dataset)
resolution = 200
root_dir = "/data/yebh/mmdet2"
dataset_type = "RSDD"
version = 1
tag = ''
for version in [1]:  # [1, 2, 3]
    config_dir = f"{root_dir}/configs/{dataset_type}/{dataset}/yolov3-{resolution}-{dataset}{tag}.py"
    work_dir = f"{root_dir}/work_dirs/{dataset_type}/{dataset}/yolov3-{resolution}-{dataset}{tag}"
    work_dir = f"{work_dir}-gpu{gpu}-v{version}"
    
    subprocess.run(f"python {root_dir}/tools/train.py "
                   f"{config_dir} --work-dir={work_dir} --gpu-id={gpu} --auto-scale-lr --seed=1079546523", shell=True)

    # work_dir = f"{work_dir}-{GPUS_num}gpu-dist-v{version}"
    # subprocess.run(f"CUDA_VISIBLE_DEVICES={GPUS} python -m torch.distributed.launch "
    #                f"--nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node={GPUS_num} "
    #                f"--master_port={PORT} {root_dir}/tools/train.py "
    #                f"{config_dir} --work-dir={work_dir} --auto-scale-lr --seed=1079546523 "
    #                f"--launcher pytorch", shell=True)

    subprocess.run(f"python {root_dir}/read_json_and_save_topk.py --path={work_dir} --gpu={GPUS[0]}", shell=True)
    subprocess.run(f"python test.py --source_dataset={source_dataset} --target_dataset {target_dataset} "
                   f"--config_dir={config_dir} "
                   f"--work_dir={work_dir} --gpu={GPUS[0]}", shell=True)
