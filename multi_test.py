import os
def multi_test(work_fdir, gpu=0):
    # gpu=1
    for file_dir in os.listdir(work_fdir):
        # test every folder
        work_dir = os.path.join(work_fdir, file_dir)
        # find config file
        file_list = os.listdir(work_dir)
        top_iter_dir=''
        config_dir=''
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

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/T2L_SCL/alpha'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/T2L_SCL/adversarial'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/T2L_SCL/at_a+b'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/T2L_SCL/cate-1k1k'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/a+b'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/adversarial'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/cate-1h1h'
# multi_test(work_fdir)

# work_fdir = '/media/data3/yebh/code/mmdetection/work_dirs/test_dataset/L2T_SCL/cate-1k1k'
# multi_test(work_fdir)

# work_fdir = '/home/yebh/mmdetection/work_dirs/test_dataset/L2T_SCL/CFA_608/ok'
# multi_test(work_fdir, gpu=0)

# work_fdir = '/home/yebh/work_dirs_remote/test_dataset/T2L_SCL_608/CFA'
# multi_test(work_fdir, gpu=6)

work_fdir = '/home/yebh/mmdetection/work_dirs/test_dataset/L2T_SCL/CFA_608/'
multi_test(work_fdir, gpu=5)

work_fdir = '/home/yebh/mmdetection/work_dirs/test_dataset/T2L_SCL/CFA_608/'
multi_test(work_fdir, gpu=5)