import json
import os
import sys

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='help demo,what writted here will be displaced in the first part of the help message.')
    parser.add_argument('--path', default='.',
                        help='the architecture of CNN, at this time we only support alexnet and vgg.')
    parser.add_argument('--gpu', default=0, help='the index of gpu.')
    args = parser.parse_args()
    return args


args = parse_args()
topK = 10
root_path = os.getcwd()
# print(root_path)
path = os.path.join(root_path, 'work_dirs')


def deal_path(path):
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
                        dict_ap50[dict_json['epoch']
                                  ] = dict_json['bbox_mAP_50']
                        dict_apall[dict_json['epoch']
                                   ] = dict_json['bbox_mAP_copypaste']

    path_result = os.path.join(
        path,  f'{path_json.split("/")[-1].split(".")[0]}.txt')

    # save result
    list_sort = sorted(dict_ap50.items(), key=lambda kv: (
        kv[1], kv[0]), reverse=True)
    with open(path_result, 'w') as load_f:
        for line in list_sort[:topK]:
            load_f.write(str(list((line[0], dict_apall[line[0]]))))
            load_f.write('\n')
            dict_top[line[0]] = line[1]

    # delete pth
    print(dict_top.keys())
    latest_checkpoint = 0
    for i in range(274, -1, -1):
        if i not in dict_top.keys():
            filename = path + f'/epoch_{i}.pth'
            if os.path.exists(filename):
                if latest_checkpoint >= 1:
                    os.remove(filename)
                else:
                    latest_checkpoint += 1

    def multi_test(work_dir, gpu=0):
        # find config file
        file_list = os.listdir(work_dir)
        top_iter_dir = ''
        config_dir = ''
        for file_name in file_list:
            if file_name.endswith('.py'):
                config_name = file_name
                config_dir = os.path.join(work_dir, file_name)
            elif file_name.endswith('txt'):
                top_iter_name = file_name
                top_iter_dir = os.path.join(work_dir, top_iter_name)
        if top_iter_dir != '':
            # read top iter and
            with open(top_iter_dir, 'r') as f:
                for i in range(topK):
                    line = f.readline().strip('\n')
                    iter = int(line.split(',')[0][1:])
                    model_dir = f"{work_dir}/epoch_{iter}.pth"
                    old_log_dir = f"{work_dir}/0_{i}th_{iter}iter.log"
                    log_dir = f'{work_dir}/0_{i}th_{iter}_APARF.log'
                    show_dir = f"{work_dir}/{iter}_image"
                    # print(log_dir)
                    if not os.path.exists(log_dir):
                        if os.path.exists(old_log_dir):
                            os.remove(old_log_dir)
                        os.system(f'CUDA_VISIBLE_DEVICES={gpu} python {root_path}/tools/test.py \
                            {config_dir} \
                            {model_dir} \
                            --eval bbox \
                            --eval-options "classwise=True" \
                            --log_dir {log_dir}')

    # test
    # multi_test(path, args.gpu)


def scan_path(fpath):
    for root, dirs, files in os.walk(fpath):
        # print(root)
        is_json = False
        is_txt = False
        is_py = False
        is_pth = False
        is_json = False
        for file in files:
            if file.endswith('.json'):
                is_json = True
            if file.endswith('.txt'):
                is_txt = True
            if file.endswith('.py'):
                is_py = True
            if file.endswith('.pth'):
                is_pth = True
        if is_json and is_txt and is_py and is_pth:
            # print(root)
            deal_path(root)


scan_path(path)
