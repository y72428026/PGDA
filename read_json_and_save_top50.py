import json, os

import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='help demo,what writted here will be displaced in the first part of the help message.')
    parser.add_argument('--path', default='.', help='the architecture of CNN, at this time we only support alexnet and vgg.')
    args = parser.parse_args() 
    return args
args = parse_args()
path_json=''

path = os.path.join('/home/yebh/mmdetection/', args.path)
# path = '/home/yebh/mmdetection/work_dirs/test_dataset/L2T_SCL/CFA_608/uda_yolov3_L2T_SCL-100e-1h1h0-0-ohem-gpu2-v0'
# find json file
# read json
list_json = []
dict_ap50 = dict()
dict_apall = dict()
dict_top = dict()

file_list = os.listdir(path)
for file_name in file_list:
    if file_name.endswith('.json'):
        path_json = os.path.join(path, file_name)
        print(file_name)
        with open(path_json, 'r') as load_f:
            for line in load_f.readlines():
                dict_json = json.loads(line)
                list_json.append(dict_json)
                if dict_json.get('mode') == 'val' and dict_json.get('bbox_mAP_50') != None:
                    dict_ap50[dict_json['epoch']] = dict_json['bbox_mAP_50']
                    dict_apall[dict_json['epoch']] = dict_json['bbox_mAP_copypaste']

path_result = os.path.join(path,  f'{path_json.split("/")[-1].split(".")[0]}.txt')

# save result
list_sort = sorted(dict_ap50.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
with open(path_result, 'w') as load_f:
    for line in list_sort[:30]:
        load_f.write(str(list((line[0],dict_apall[line[0]]))))
        load_f.write('\n')
        dict_top[line[0]] = line[1]

# delete pth
print(dict_top.keys()) 
latest_checkpoint = 0
for i in range(274,-1,-1):
    if i not in dict_top.keys(): 
        filename = path + f'/epoch_{i}.pth'
        if os.path.exists(filename):
            if latest_checkpoint>=3:
                os.remove(filename)
            else:
                latest_checkpoint += 1




