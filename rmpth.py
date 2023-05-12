# walk the path, if the number of pth in the dir is more than 15, execute the following code
import os, shutil, time

<<<<<<< Updated upstream
<<<<<<< Updated upstream
scan_path = '/media/data3/yebh/code/mmdet2/work_dirs'
=======
scan_path = '/home/yebh/mmdetection/work_dirs'
>>>>>>> Stashed changes
=======
scan_path = '/home/yebh/mmdetection/work_dirs'
>>>>>>> Stashed changes
while 1:
    # print time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for root, dirs, files in os.walk(scan_path):
        pth_num = len([file for file in files if file.endswith('.pth')])
        if pth_num > 20+1+1:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            # gpu = int(root.split('-gpu')[-1][0])
            print(root, ' start_pth_num: ', pth_num)
            # print(pth_num)
            # input('go')
            os.system(f'python read_json_and_save_top20.py --path={root}') 
            # input('continue?')
    # sleep 30min
    print('sleep 30min')
    time.sleep(30*60)
=======
=======
>>>>>>> Stashed changes
            gpu = int(root.split('-gpu')[-1][0])
            print(root, ' start_pth_num: ', pth_num)
            # print(pth_num)
            # input('go')
            os.system(f'python read_json_and_save_top20.py --path={root} --gpu={gpu}') 
            # input('continue?')
    # sleep 30min
    time.sleep(60*30)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
