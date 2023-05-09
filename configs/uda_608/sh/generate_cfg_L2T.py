import os
root_path = '/media/data3/yebh/code/mmdet2/configs/uda_608/sh'
template_name = 'uda_yolov3_L2T_SCL_template.py'
template_path = os.path.join(root_path,template_name)

dataset='L2T'
# GFA weight
DA='1h00'
DA_w0=100
DA_w1=0
DA_w2=0

# CFA weight
cfa_weight=0.0875
cfg_v_list=[9,11,12,13,14]
# T_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# a_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
T_list=[0.5]
a_list=[0.1,1]

# if use fp16
fp16='-fp16'
# fp16=''

new_cfg_path = os.path.join(root_path,dataset)
if not os.path.exists(new_cfg_path):
    os.makedirs(new_cfg_path, exist_ok=True)

# generate cfg
for T in T_list:
    for a in a_list:
        for cfg_v in cfg_v_list:
            # name
            new_cfg_name = f'uda-yolov3-{dataset}-SCL-Xohem-{DA}-cfav{cfg_v}-{cfa_weight}-T{T}-a{a}{fp16}.py'
            
            print(new_cfg_name)
            with open(template_path, "r") as f:
                srcCfg = f.read().split("\n")
                # fp16
                srcCfg[1-1] = f"_base_ = [  '../../../yolo/uda_yolov3_608_{dataset}_SCL_Xohem-100e-000-0{fp16}.py', ]"
                # GFA weight
                srcCfg[6-1] =f'        img_weight={DA_w0},'
                srcCfg[9-1] =f'        img_weight={DA_w1},'
                srcCfg[12-1]=f'        img_weight={DA_w2},'
                # CFA weight
                srcCfg[14-1] =f'    cfa_thres={T},'
                srcCfg[15-1] =f'    enable_category_loss=True,'
                srcCfg[16-1] =f'    category_weight={cfa_weight},'
                srcCfg[17-1] =f'    alpha_mb={a},'
                srcCfg[18-1] =f'    cfa_v={cfg_v},'
                f.close()
                with open(os.path.join(new_cfg_path, new_cfg_name), "w") as f:
                    f.write('\n'.join(srcCfg))
            # input('check')