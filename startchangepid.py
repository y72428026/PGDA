import os, shutil, time

while 1:
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    os.system('python ./changepid.py')
    print('sleep 15min')
    time.sleep(60*15)