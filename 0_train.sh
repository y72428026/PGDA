gpu=0

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-000-0.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-000-0
work_dir=${work_dir}-gpu${gpu}-v1
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-000-0001-T05-a1.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-000-0001-T05-a1-cfav11-ohem
work_dir=${work_dir}-gpu${gpu}-v1
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-000-0001-T05-a1.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-000-0001-T05-a1-cfav11-ohem
work_dir=${work_dir}-gpu${gpu}-v2
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-000-0001-T05-a1.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-000-0001-T05-a1-cfav11-ohem
work_dir=${work_dir}-gpu${gpu}-v3
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-000-001-T05-a1.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-000-001-T05-a1-cfav11-ohem
work_dir=${work_dir}-gpu${gpu}-v1
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-000-01-T05-a1.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-000-01-T05-a1-cfav11-ohem
work_dir=${work_dir}-gpu${gpu}-v1
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}







