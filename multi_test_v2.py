import os
def multi_test(work_fdir, gpu=0):
    # walk work_fdir to find all folder
    for root, dirs, files in os.walk(work_fdir):
        is_py = False
        is_txt = False
        is_json = False
        is_pth = False
        is_finish = False
        for file in files:
            if file.endswith('.py'):
                is_py = True
            if file.endswith('.txt'):
                is_txt = True
            if file.endswith('.json'):
                is_json = True
            if file.endswith('.pth'):
                is_pth = True
                if file ==  'epoch_273.pth' and 'T2L_SCL' in root:
                    is_finish = True
                if file ==  'epoch_100.pth' and 'L2T_SCL' in root:
                    is_finish = True
        if is_py and is_txt and is_json and is_pth and is_finish:
            work_dir = root
            # find config file
            file_list = files
            top_iter_dir=''
            config_dir=''
            for file_name in file_list:
                if file_name.endswith('.py'):
                    config_dir = os.path.join(work_dir, file_name)
                elif file_name.endswith('txt'):
                    top_iter_name = file_name
                    top_iter_dir = os.path.join(work_dir, top_iter_name)
            # print(f'work_dir: {root}\n, cfg: {config_dir}\n, top_iter_txt: {top_iter_dir}\n')
            # input('continue')
            if top_iter_dir != '':
                # read top iter and 
                with open(top_iter_dir, 'r') as f:
                    for i in range(10):
                        line = f.readline().strip('\n')
                        iter = int(line.split(',')[0][1:])
                        model_dir=f"{work_dir}/epoch_{iter}.pth"
                        old_log_dir=f"{work_dir}/0_{i}th_{iter}iter.log"
                        log_dir=f'{work_dir}/0_{i}th_{iter}_APARF.log'
                        show_dir=f"{work_dir}/{iter}_image"
                        # print(log_dir)
                        if not os.path.exists(log_dir):
                            if os.path.exists(old_log_dir):
                                os.remove(old_log_dir)
                            os.system(f'CUDA_VISIBLE_DEVICES={gpu} python tools/test.py \
                                {config_dir} \
                                {model_dir} \
                                --eval bbox \
                                --eval-options "classwise=True" \
                                --log_dir {log_dir}')

# work_fdir = '/home/yebh/mmdetection/work_dirs/test_dataset/T2L_SCL'
# work_fdir = '/home/yebh/work_dirs_remote/test_dataset/L2T_SCL_608'
# multi_test(work_fdir, gpu=7)
# work_fdir = '/home/yebh/work_dirs_remote/test_dataset/T2L_SCL_608'
work_fdir = '/home/yebh/mmdetection/work_dirs/test_dataset/L2T_SCL'
work_fdir = '/home/yebh/work_dirs_remote/DAOD/T2L_SCL'
# work_fdir = '/home/yebh/mmdetection/work_dirs/test_dataset/T2L_SCL/CFA_608/uda-yolov3-T2L-SCL-75k75k0-cfav9-0.075-cT0.3-pT0.7-a0.1-gpu2-v1'
multi_test(work_fdir, gpu=6)