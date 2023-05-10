gpu=6

### T2L

dataset=T2L
cfav=9
conf_T=0.3
pred_T=0.3
a=0.7
DA=75k75k0
cfa_weight=0.0625
# fp16=-fp16

for version in 1 2 3
do
    configs=./configs/uda_608/uda_yolov3_${dataset}_SCL_${DA}-cfav${cfav}-${cfa_weight}-cT${T}-pT${T}-a${a}${fp16}.py
    work_dir=./work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda_yolov3_${dataset}_SCL_${DA}-${cfa_weight}-T${T}-a${a}-cfa${cfg}${fp16}
    work_dir=${work_dir}-gpu${gpu}-v$version
    python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
    python read_json_and_save_top50.py --path=${work_dir} 
done


### L2T

# dataset=L2T
# cfg=9
# T=0.5
# a=0.1
# DA=1h00
# cfa_weight=0.0875
# fp16=-fp16

# for version in 1 2 3 4
# do
#     configs=./configs/uda_608/sh/L2T/uda-yolov3-${dataset}-SCL-Xohem-${DA}-cfav${cfg}-${cfa_weight}-T${T}-a${a}${fp16}.py
#     work_dir=./work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda-yolov3-${dataset}-SCL-Xohem-${DA}-cfav${cfg}-${cfa_weight}-T${T}-a${a}${fp16}
#     work_dir=${work_dir}-gpu${gpu}-v$version
#     python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
#     python read_json_and_save_top50.py --path=${work_dir} 
# done