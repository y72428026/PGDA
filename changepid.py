import os
import random
import subprocess

def get_user_processes(user):
    try:
        output = subprocess.check_output(['ps', '-u', user, '-o', 'pid,comm'], universal_newlines=True)
        lines = output.strip().split('\n')[1:]
        processes = [line.split() for line in lines]
        return processes
    except subprocess.CalledProcessError:
        return []

def get_cpu_affinity(pid):
    try:
        output = subprocess.check_output(['taskset', '-pc', str(pid)], universal_newlines=True)
        affinity = output.strip().split(': ')[-1]
        return affinity
    except subprocess.CalledProcessError:
        return ''

def set_cpu_affinity(pid, cpu_core):
    try:
        subprocess.check_call(['taskset', '-pc', cpu_core, str(pid)])
    except subprocess.CalledProcessError:
        pass

def is_valid_cpu_affinity(cpu_affinity):
    # Check if CPU affinity includes only cores 0 and 40
    return set(cpu_affinity.split(',')) == {'0', '40'}

user = 'yebh'

import os, shutil, time

# while 1:
processes = get_user_processes(user)
# print time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for process in processes:
    pid = process[0]
    cpu_affinity = get_cpu_affinity(pid)
    if pid == '1137413':
        print(f'cpu_affinity: {cpu_affinity}')
        input('is_valid_cpu_affinity')
    if cpu_affinity and is_valid_cpu_affinity(cpu_affinity):
        # Generate a new random CPU core
        new_cpu_core_1 = random.randint(0, 40)
        while True:
            new_cpu_core_2 = random.randint(0, 40)
            if new_cpu_core_2 != new_cpu_core_1:
                break
        new_cpu_core = f"{new_cpu_core_1},{new_cpu_core_2}"
        set_cpu_affinity(pid, new_cpu_core)
        print(f"Changed CPU core affinity for process {pid} to {new_cpu_core}")
    # else:
    #     print(f"Skipped changing CPU core affinity for process {pid}")

# sleep 15min
# print('sleep 15min')
# time.sleep(60*15)