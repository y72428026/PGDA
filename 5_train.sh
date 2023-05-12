gpu=5

### T2L
dataset=T2L
cfav=9
conf_T=0.3
# pred_T=0.3
a=0.7
DA=75k75k0
cfa_weight=0.0625
# fp16=-fp16
# for version in 1 2 3
# do
#     for pred_T in 0.2 0.3  
#     # for pred_T in 0.5 0.6 0.7 0.8 0.9
#     do
#         configs=configs/uda_608/shv2/${dataset}/uda-yolov3-${dataset}-SCL-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}.py
#         work_dir=work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda-yolov3-${dataset}-SCL-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}
#         work_dir=${work_dir}-gpu${gpu}-v$version
#         python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
#         python read_json_and_save_top50.py --path=${work_dir} 
#     done
# done

configs=work_dirs/test_dataset/T2L_SCL/CFA_608/uda-yolov3-T2L-SCL-75k75k0-cfav9-0.0625-cT0.3-pT0.2-a0.7-gpu5-v2/uda-yolov3-T2L-SCL-75k75k0-cfav9-0.0625-cT0.3-pT0.2-a0.7.py
work_dir=work_dirs/test_dataset/T2L_SCL/CFA_608/uda-yolov3-T2L-SCL-75k75k0-cfav9-0.0625-cT0.3-pT0.2-a0.7-gpu5-v2
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir} 


# ### L2T
# dataset=L2T
# DA=1h00
# cfa_v=14
# cfa_weight=0.0875
# T=0.5
# a=1
# fp16=-fp16
# for version in 1 2 3 
# do
#     configs=./configs/uda_608/sh/L2T/uda-yolov3-${dataset}-SCL-Xohem-${DA}-cfav${cfa_v}-${cfa_weight}-T${T}-a${a}${fp16}.py
#     work_dir=./work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda-yolov3-${dataset}-SCL-Xohem-${DA}-cfav${cfa_v}-${cfa_weight}-T${T}-a${a}${fp16}
#     work_dir=${work_dir}-gpu${gpu}-v$version
#     python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
#     python read_json_and_save_top50.py --path=${work_dir} 
# done