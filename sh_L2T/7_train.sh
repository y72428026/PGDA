gpu=7

### L2T
dataset=L2T
cfav=9
# conf_T=0.5
pred_T=0.1
a=0.8
DA=1h1h0
cfa_weight=0.0875

for version in 1 2 3
do
    for conf_T in 0.0, 0.1
    do
        configs=configs/uda_608/shv2/${dataset}/uda-yolov3-${dataset}-SCL-Xohem-100e-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}.py
        work_dir=work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda-yolov3-${dataset}-SCL-Xohem-100e-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}
        work_dir=${work_dir}-gpu${gpu}-v$version
        python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
        python read_json_and_save_top50.py --path=${work_dir} 
    done
done

# ### T2L
# dataset=T2L
# cfav=9
# conf_T=0.3
# # pred_T=0.3
# a=0.7
# DA=75k75k0
# cfa_weight=0.0625
# # fp16=-fp16
# for version in 1 2 3
# do
#     for pred_T in 0.7 0.8 0.9
#     do
#         configs=configs/uda_608/shv2/${dataset}/uda-yolov3-${dataset}-SCL-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}.py
#         work_dir=work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda-yolov3-${dataset}-SCL-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}
#         work_dir=${work_dir}-gpu${gpu}-v$version
#         python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
#         python read_json_and_save_top50.py --path=${work_dir} 
#     done
# done





 