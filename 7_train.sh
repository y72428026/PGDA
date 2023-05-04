gpu=7

# configs=/home/yebh/mmdetection/configs/uda_608/fix/uda_yolov3_T2L_75k75k0-0.py
# work_dir=work_dirs/test_dataset/T2L_SCL/CFA_608/uda_yolov3_T2L_SCL_75k75k0-0-T03-a07-visual
# work_dir=${work_dir}-gpu${gpu}-v0
# python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523
# python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/fix/uda_yolov3_L2T_SCL_1h1h0-00875-T05-a01.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL_1h1h0-00875-T05-a01
work_dir=${work_dir}-gpu${gpu}-vx
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/fix/uda_yolov3_L2T_SCL_1h1h0-00875-T05-a1.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL_1h1h0-00875-T05-a1
work_dir=${work_dir}-gpu${gpu}-vx
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}
