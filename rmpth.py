# walk the path, if the number of pth in the dir is more than 15, execute the following code
import os
import shutil
import time
from read_json_and_save_topk_scan import deal_path

path = os.getcwd()
scan_path = os.path.join(path, 'work_dirs')
print(f'scan path: {scan_path}')
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
            # os.system(
            #     f'python read_json_and_save_topk_scan.py --path={root} --gpu={gpu}')
            deal_path(root)
    # sleep 15min
    print('sleep 10min')
    time.sleep(60*10)
