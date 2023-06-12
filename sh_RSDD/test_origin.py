import glob, os

train_dataset = "RSDDII2RSDDI"
test_datasets = ["RSDDI", "RSDDII"]  # Add more test datasets if needed
dataset_type = "RSDD"
resolution = 200
model_tag = "UDA"

cfav=9
conf_T=0.3
pred_T=0.7
a=0.1
DA="75k75k0"
cfa_weight=0.0625
fp16=''

root_dir = "/data/yebh"
mmdet2_dir = f"{root_dir}/mmdet2"
# work_dir = f"{mmdet2_dir}/work_dirs/{dataset_type}/{model_tag}/yolov3-{resolution}-{train_dataset}*"
work_dir = f"{mmdet2_dir}/work_dirs/{dataset_type}/{train_dataset}/yolov3-{model_tag}-{resolution}-{train_dataset}-{DA}-cfav{cfav}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}*"
work_dir = glob.glob(work_dir)[0]
print(work_dir)

txt_dir = glob.glob(f"{work_dir}/*.txt")[0]
print(txt_dir)
with open(txt_dir, "r") as file:
    line = file.readline()
epoch = line.split(",")[0][1:]
model_dir = f"{work_dir}/epoch_{epoch}.pth"
print(model_dir)
for test_dataset in test_datasets:
    # config_dir = f"{work_dir}/yolov3-{resolution}-{test_dataset}.py"
    config_dir = f"{mmdet2_dir}/configs/{dataset_type}/{train_dataset}/yolov3-{model_tag}-{resolution}-{train_dataset}-{DA}-cfav{cfav}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}.py"
    # config_dir = f"{mmdet2_dir}/configs/{dataset_type}/{model_tag}/yolov3-{resolution}-{test_dataset}.py"
    log_dir = f"{work_dir}/_{epoch}_CM/_{epoch}_{test_dataset}_APARF.log"
    out_dir = f"{work_dir}/_{epoch}_CM/_{epoch}_{test_dataset}.pkl"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    print(log_dir)
    print(out_dir)
    gpu_ids = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

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
        f"data.test.img_prefix={root_dir}/dataset/{test_dataset}/JPEGImages",
        f"data.test.ann_file={root_dir}/dataset/{test_dataset}/{test_dataset}_test.json"
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
        f"data.test.img_prefix={root_dir}/dataset/{test_dataset}/JPEGImages",
        f"data.test.ann_file={root_dir}/dataset/{test_dataset}/{test_dataset}_test.json"
    ])
