# walk the path, delete file with .pyc suffix
# Usage: python rmpyc.py

import os
import sys
path = '/home/yebh/mmdetection'
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith('.pyc'):
            print('Delete %s ?' % os.path.join(root, name))
            os.remove(os.path.join(root, name))