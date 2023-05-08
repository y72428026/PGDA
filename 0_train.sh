gpu=0
cfg=9
T=05
a=1
for a in 01 1
do
    for version in 1 2 3
    do
        configs=./configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-1h1h0-00875-T${T}-a${a}-cfa${cfg}.py
        work_dir=./work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-00875-T${T}-a${a}-cfav${cfg}-ohem
        work_dir=${work_dir}-gpu${gpu}-v$version
        python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
        python read_json_and_save_top50.py --path=${work_dir} 
    done
done









