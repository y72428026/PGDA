import subprocess
import random
import os, glob
# Set the "-e" option
subprocess.run("set -e", shell=True)

# Set variables
GPUS_num = 4
gpu = 1
GPUS = "0,1,2,3"
PORT = random.randint(29500, 29599)


source_dataset = 'RSDDIICrop1Class'
target_dataset = 'RSDDIICropAdjust1Class'
dataset_type = "RSDD"

dataset = source_dataset
resolution = 640
root_dir = os.getcwd().split('mmdet2')[0] + 'mmdet2'

GFAweight = 426
CFAwegiht = 0.01
CFAconfT = 0.7
CFApredT = 0.7
CFAalpha = 0.1

for version in [1]:
    config_dir = f"{root_dir}/configs/{dataset_type}/{source_dataset}2{target_dataset}/yolov3-UDA-{resolution}-{source_dataset}2{target_dataset}*.py"
    config_dir = glob.glob(config_dir)[0]
    work_dir = f"{root_dir}/work_dirs/{dataset_type}/{source_dataset}2{target_dataset}/{resolution}-{GFAweight}-{CFAwegiht}-{CFAconfT}-{CFApredT}-{CFAalpha}"
    work_dir = f"{work_dir}-v{version}"
    subprocess.run( f"python {root_dir}/tools/train.py "
                    f"{config_dir} --work-dir={work_dir} --gpu-id={gpu} --auto-scale-lr --seed=1079546523 "
                    f"--cfg-options " 
                    f"uda.da_backbone_head.img_weight={GFAweight} "
                    f"uda.da_neck_head.img_weight={GFAweight} "
                    f"uda.da_pred_head.img_weight={GFAweight} "
                    f"uda.enable_category_loss=True "
                    f"uda.category_weight={CFAwegiht} "
                    f"uda.cfa_conf_thres={CFAconfT} "
                    f"uda.cfa_pred_thres={CFApredT} "
                    f"uda.alpha_mb={CFAalpha} "
                    f"uda.cfa_v=9 "
                   , shell=True)
