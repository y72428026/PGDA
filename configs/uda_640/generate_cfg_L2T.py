import os, sys

# root_path = os.getcwd()
root_path = sys.path[0]



dataset='L2T'
dataset_tag='3class'
model_tag='UDA-SCL'
resolution=640
template_name = f'yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-template.py'
print(template_name)
template_path = os.path.join(root_path,template_name)

# GFA weight
DA='1h1h0'
DA_w0=100
DA_w1=100
DA_w2=0
# DA='25t25t0'
# DA_w0=25
# DA_w1=25
# DA_w2=0

# CFA weight
cfa_weight_list=[0.0875]
# cfa_weight_list=[0.1, 0.075, 0.0625, 0.05, 0.0375, 0.0125]
cfg_v_list=[9]
# [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
conf_T_list=[0.5]
# conf_T_list=[0.5]
pred_T_list=[0.6]
# a_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
a_list=[0.1]

# if use fp16
# fp16='-fp16'
fp16=''

new_cfg_path = os.path.join(root_path,dataset)
if not os.path.exists(new_cfg_path):
    os.makedirs(new_cfg_path, exist_ok=True)

# generate cfg
for conf_T in conf_T_list:
    for pred_T in pred_T_list:
        for a in a_list:
            for cfg_v in cfg_v_list:
                for cfa_weight in cfa_weight_list:
                    # name
                    # new_cfg_name = f'yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-100e-{DA}-cfav{cfg_v}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}.py'
                    new_cfg_name = f'yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-{DA}-cfav{cfg_v}-{cfa_weight}-cT{conf_T}-pT{pred_T}-a{a}{fp16}.py'
                    print(new_cfg_name)
                    with open(template_path, "r") as f:
                        srcCfg = f.read().split("\n")
                        # fp16
                        srcCfg[1-1] = f"_base_ = [  '../../yolo/yolov3-{model_tag}-{resolution}-{dataset}-{dataset_tag}-000-0{fp16}.py']"
                        # GFA weight
                        srcCfg[6-1] =f'        img_weight={DA_w0},'
                        srcCfg[9-1] =f'        img_weight={DA_w1},'
                        srcCfg[12-1]=f'        img_weight={DA_w2},'
                        # CFA weight
                        srcCfg[14-1] =f'    cfa_conf_thres={conf_T},'
                        srcCfg[15-1] =f'    cfa_pred_thres={pred_T},'
                        srcCfg[16-1] =f'    enable_category_loss=True,'
                        srcCfg[17-1] =f'    category_weight={cfa_weight},'
                        srcCfg[18-1] =f'    alpha_mb={a},'
                        srcCfg[19-1] =f'    cfa_v={cfg_v},'
                        f.close()
                        with open(os.path.join(new_cfg_path, new_cfg_name), "w") as f:
                            f.write('\n'.join(srcCfg))
                    # input('check')