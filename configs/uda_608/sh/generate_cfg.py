import os
root_path = '/home/yebh/mmdetection/configs/uda_608/sh'
template_name = 'uda_yolov3_T2L_SCL_tempalete.py'
template_path = os.path.join(root_path,template_name)
dataset='T2L'

# GFA weight
DA='75k75k0'
DA_w0=7500
DA_w1=7500
DA_w2=0

# CFA weight
cfa_weight=0.0625
cfg_list=[9,11,12,13,14]
T_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
a_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# if use fp16
fp16='-fp16'
# fp16=''

# generate cfg
for T in T_list:
    for a in a_list:
        for cfg in cfg_list:
            # name
            new_cfg_name = f'uda_yolov3-{dataset}-SCL-{DA}-cfav{cfg}-{cfa_weight}-T{T}-a{a}{fp16}.py'
            new_cfg_path = os.path.join(root_path,new_cfg_name)
            print(new_cfg_name)
            with open(template_path, "r") as f:
                srcCfg = f.read().split("\n")
                # fp16
                srcCfg[1-1] = srcCfg[0].replace('000-0.py',f'000-0{fp16}.py')
                # GFA weight
                srcCfg[6-1] =f'        img_weight={DA_w0},'
                srcCfg[9-1] =f'        img_weight={DA_w1},'
                srcCfg[12-1] =f'        img_weight={DA_w2},'
                # CFA weight
                srcCfg[14-1] =f'    cfa_thres={T},'
                srcCfg[16-1] =f'    category_weight={cfa_weight},'
                srcCfg[17-1] =f'    alpha_mb={a},'
                f.close()
                with open(new_cfg_path, "w") as f:
                    f.write('\n'.join(srcCfg))
            input('check')