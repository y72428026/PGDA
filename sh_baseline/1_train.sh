gpu=1

configs=../configs/uda_608/baseline/yolov3_608_L-000-0.py
work_dir=../work_dirs/baseline/L
work_dir=${work_dir}-gpu${gpu}-v$version
python ../tools/train.py $configs --gpu-id=${gpu} --work-dir=${work_dir} --auto-scale-lr --seed=1079546523 
python ../read_json_and_save_topk.py --path=${work_dir} 