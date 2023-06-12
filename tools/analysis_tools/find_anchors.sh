# python optimize_anchors.py \
#   /data/yebh/mmdet2/configs/PCB/baseline/yolov3-640-DeepPCB.py \
#   --algorithm differential_evolution  --output-dir ./DeepPCB

# python optimize_anchors.py \
#   /data/yebh/mmdet2/configs/PCB/baseline/yolov3-640-PCBCrop.py \
#   --algorithm differential_evolution  --output-dir ./PCBCrop

dataset_type=BIS
dataset_name=XQXY
# resolution=640
for resolution in 640
do
  python optimize_anchors.py \
    /data/yebh/mmdet2/configs/${dataset_type}/${dataset_name}/yolov3-${resolution}-${dataset_name}.py \
    --input-shape 640 480\
    --algorithm differential_evolution  --output-dir ./${dataset_name}_$resolution
done