import glob
import os
import argparse


def parse_args():
    args = argparse.ArgumentParser(description='Train a detector')
    args.add_argument("--source_dataset", type=str, default="RSDDII")
    args.add_argument("--target_dataset", nargs='+')
    args.add_argument("--config_dir", type=str, default="")
    args.add_argument("--dataset_tag", type=str, default="")
    args.add_argument("--work_dir", type=str, default="")
    args.add_argument("--gpu", type=int, default=0)
    return args.parse_args()


args = parse_args()

# root_dir = "/data/yebh"
root_dir = os.getcwd().split("yebh")[0] + 'yebh'
mmdet2_dir = f"{root_dir}/mmdet2"
work_dir = os.path.join(mmdet2_dir, args.work_dir)
config_dir = os.path.join(mmdet2_dir, args.config_dir)
# input(config_dir)
dataset_tag = args.dataset_tag
print(f'dataset_tag: {dataset_tag}')
# train_dataset = f"{args.source_dataset}2{args.target_dataset}"
args.target_dataset.append(args.source_dataset)
args.target_dataset = list(set(args.target_dataset))
print(work_dir)
txt_dir = glob.glob(f"{work_dir}/*.txt")[0]
with open(txt_dir, "r") as file:
    line = file.readline()
epoch = line.split(",")[0][1:]
model_dir = f"{work_dir}/epoch_{epoch}.pth"

for test_dataset in args.target_dataset:
    log_dir = f"{work_dir}/_{epoch}_CM/_{epoch}_{test_dataset}_APARF.log"
    out_dir = f"{work_dir}/_{epoch}_CM/_{epoch}_{test_dataset}.pkl"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import subprocess
    subprocess.call([
        "python",
        f"{mmdet2_dir}/tools/test.py",
        config_dir,
        model_dir,
        "--eval",
        "bbox",
        "--eval-options",
        "classwise=True",
        "--log_dir",
        log_dir,
        "--out",
        out_dir,
        "--cfg-options",
        f"data.test.img_prefix={root_dir}/dataset/{dataset_tag}/{test_dataset}/JPEGImages",
        f"data.test.ann_file={root_dir}/dataset/{dataset_tag}/{test_dataset}/{test_dataset}_test.json"
    ])

    subprocess.call([
        "python",
        f"{mmdet2_dir}/tools/analysis_tools/confusion_matrix.py",
        config_dir,
        out_dir,
        work_dir,
        "--tag",
        f"_{epoch}_CM/_{epoch}_{test_dataset}",
        "--cfg-options",
        f"data.test.img_prefix={root_dir}/dataset/{dataset_tag}/{test_dataset}/JPEGImages",
        f"data.test.ann_file={root_dir}/dataset/{dataset_tag}/{test_dataset}/{test_dataset}_test.json"
    ])
