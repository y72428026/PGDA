gpu=4

configs=/home/yebh/mmdetection/configs/uda_608/uda_yolov3_T2L_SCL_75k75k0-00625-T03-a07.py
work_dir=work_dirs/test_dataset/T2L_SCL/CFA_608/uda_yolov3_T2L_SCL_75k75k0-00625-T03-a07-cfav11
work_dir=${work_dir}-gpu${gpu}-v3
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/uda_yolov3_T2L_SCL_75k75k0-00625-T03-a07.py
work_dir=work_dirs/test_dataset/T2L_SCL/CFA_608/uda_yolov3_T2L_SCL_75k75k0-00625-T03-a07-cfav11
work_dir=${work_dir}-gpu${gpu}-v1
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/uda_yolov3_T2L_SCL_75k75k0-00625-T03-a07.py
work_dir=work_dirs/test_dataset/T2L_SCL/CFA_608/uda_yolov3_T2L_SCL_75k75k0-00625-T03-a07-cfav11
work_dir=${work_dir}-gpu${gpu}-v2
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523
python read_json_and_save_top50.py --path=${work_dir}
