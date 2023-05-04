gpu=2

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-1t1t0-0.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-1t1t0-0-ohem
work_dir=${work_dir}-gpu${gpu}-v0
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 --auto-resume
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-1k1k0-0.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-1k1k0-0-ohem
work_dir=${work_dir}-gpu${gpu}-v0
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-330-0.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-330-0-ohem
work_dir=${work_dir}-gpu${gpu}-v0
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-3t3t0-0.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-3t3t0-0-ohem
work_dir=${work_dir}-gpu${gpu}-v0
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-3h3h0-0.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-3h3h0-0-ohem
work_dir=${work_dir}-gpu${gpu}-v0
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}
# t03
configs=/home/yebh/mmdetection/configs/uda_608/100e/uda_yolov3_L2T_SCL-100e-110-0001-T03-a1-cfa11.py
work_dir=work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-110-0001-T03-a1-cfav11-ohem
work_dir=${work_dir}-gpu${gpu}-v0
python tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python read_json_and_save_top50.py --path=${work_dir}

