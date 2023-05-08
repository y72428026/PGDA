gpu=4

dataset=T2L
cfg=13
T=03
a=07
DA=75k00
cfa_weight=00625
fp16=-fp16

for version in 1 2 3
do
    for DA in 75k75k0 75k00
    do
        configs=./configs/uda_608/uda_yolov3_${dataset}_SCL_${DA}-${cfa_weight}-T${T}-a${a}-cfa${cfg}${fp16}.py
        work_dir=./work_dirs/test_dataset/${dataset}_SCL/CFA_608/uda_yolov3_${dataset}_SCL_${DA}-${cfa_weight}-T${T}-a${a}-cfa${cfg}${fp16}
        work_dir=${work_dir}-gpu${gpu}-v$version
        python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
        python read_json_and_save_top50.py --path=${work_dir} 
    done
done
