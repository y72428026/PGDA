# walk the path, if the number of pth in the dir is more than 15, execute the following code
import os, shutil, time


scan_path = '/home/yebh/mmdet2/work_dirs'
while 1:
    # print time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for root, dirs, files in os.walk(scan_path):
        pth_num = len([file for file in files if file.endswith('.pth')])
        if pth_num > 20+1+1:
            try:
                gpu = int(root.split('-gpu')[-1][0])
            except:
                gpu = 7
            print(root, ' start_pth_num: ', pth_num)
            os.system(f'python read_json_and_save_top20.py --path={root} --gpu={gpu}') 
    # sleep 15min
    print('sleep 15min')
    time.sleep(60*15)
