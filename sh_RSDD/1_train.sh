gpu=1

dataset=PCBCrop2DeepPCB
model_tag=UDA
resolution=640

cfav=9
# conf_T=0.5
pred_T=0.6
a=0.1
DA=1h1h0
cfa_weight=0.0875
dataset_type=PCB
for version in 1 # 2 3
do
    for conf_T in 0.5
    do
        configs=../configs/${dataset_type}/${dataset}/yolov3-${model_tag}-${resolution}-${dataset}-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}.py
        work_dir=../work_dirs/${dataset_type}/${dataset}/yolov3-${model_tag}-${resolution}-${dataset}-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}
        work_dir=${work_dir}-gpu${gpu}-v$version
        python ../tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
        # python ../read_json_and_save_topk.py --path=${work_dir} 
    done
done









