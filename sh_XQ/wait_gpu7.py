import os, sys, time
import subprocess, pynvml
def choose_available_gpu(gpu=3) -> int:
    # judge the status of GPU : if GPU is available, 
    # then scan the queue( can by folder json etc.) and run the job, return the result to user
    percent_not_used = 0.10 # 10% of GPU memory is not used
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    # print("GPU count: %d" % count)

    i = gpu
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # print("GPU used memory: %d" % info.used)
    # print("GPU all memory: %d" % info.total)
    # print("GPU free memory: %d" % info.free)
    percent_used = float(info.used) / float(info.total) * 100
    print('GPU %d used memory percent: %f' % (i, percent_used))
    if float(info.used) / float(info.total) < percent_not_used:
        pynvml.nvmlShutdown()
        print("GPU %i is available" % i)
        # input("Press Enter to continue...")
        return i
    pynvml.nvmlShutdown()
    print("GPU is not available")
    return None

gpu = 7
while True:
    if choose_available_gpu(gpu):
        time.sleep(10*60)
        if choose_available_gpu(gpu):
            os.system('python train_uda.py')
            break
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('wait')
        # sleep 10 minutes
        time.sleep(10*60)
        