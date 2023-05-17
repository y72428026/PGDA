gpu=4

### T2L
dataset=T2L
cfav=9
conf_T=0.3
pred_T=0.7
a=0.1
DA=75k75k0
# cfa_weight=0.0625
for version in 1 2 3
do
    for cfa_weight in 0.0875 0.025
    do
        configs=../configs/uda_608/shv2/${dataset}/uda-yolov3-${dataset}-SCL-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}.py
        work_dir=../work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda-yolov3-${dataset}-SCL-${DA}-cfav${cfav}-${cfa_weight}-cT${conf_T}-pT${pred_T}-a${a}${fp16}
        work_dir=${work_dir}-gpu${gpu}-v$version
        python ../tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
        python ../read_json_and_save_top50.py --path=${work_dir} 
    done
done
