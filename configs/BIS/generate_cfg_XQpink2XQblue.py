import os
import sys
root_path = sys.path[0]

dataset = 'XQpink2XQblue'
dataset_tag = '5class'
model_tag = 'UDA'
resolution = 640
template_name = f'yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-template.py'
template_path = os.path.join(root_path, template_name)

# GFA weight
DA_w0 = 462
DA_w1 = DA_w0
DA_w2 = DA_w0
DA_w3 = 0
# CFA weight
cfa_weight_list = [0.01]
# cfa_weight_list=[0.05, 0.075, 0.0875, 0.0375, 0.025, 0.0125, 0.001]
cfg_v_list = [9]
# [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
conf_T_list = [0.4]
# conf_T_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
pred_T_list = [0.4]
a_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# a_list = [0.1]

# if use fp16
# fp16='-fp16'
fp16 = ''

new_cfg_path = os.path.join(root_path, dataset)
if not os.path.exists(new_cfg_path):
    os.makedirs(new_cfg_path, exist_ok=True)

# generate cfg
for conf_T in conf_T_list:
    for pred_T in pred_T_list:
        pred_T = conf_T
        for a in a_list:
            for cfg_v in cfg_v_list:
                for cfa_weight in cfa_weight_list:
                    # name
                    # new_cfg_name = f'uda-yolov3-{dataset}-SCL-{DA}-cfav{cfg_v}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}.py'
                    new_cfg_name = f'yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-DA-{DA_w0}-{DA_w1}-{DA_w2}-{DA_w3}-cfav{cfg_v}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}.py'
                    print(new_cfg_name)
                    with open(template_path, "r") as f:
                        srcCfg = f.read().split("\n")
                        # fp16
                        srcCfg[1 -
                            1] = f"_base_ = [  '../../yolo/yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-0000-0{fp16}.py']"
                        # GFA weight
                        srcCfg[5-1] = f'        img_weight={DA_w0},'
                        srcCfg[8-1] = f'        img_weight={DA_w1},'
                        srcCfg[11-1] = f'        img_weight={DA_w2},'
                        srcCfg[14-1] = f'        img_weight={DA_w3},'
                        # CFA weight
                        srcCfg[16-1] = f'    cfa_conf_thres={conf_T},'
                        srcCfg[17-1] = f'    cfa_pred_thres={pred_T},'
                        srcCfg[18 -
                            1] = f'    enable_category_loss={cfa_weight>0},'
                        srcCfg[19-1] = f'    category_weight={cfa_weight},'
                        srcCfg[20-1] = f'    alpha_mb={a},'
                        srcCfg[21-1] = f'    cfa_v={cfg_v},'
                        f.close()
                        with open(os.path.join(new_cfg_path, new_cfg_name), "w") as f:
                            f.write('\n'.join(srcCfg))

