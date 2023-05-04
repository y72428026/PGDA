# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS, build_dataset
from .custom import CustomDataset
from .coco import CocoDataset

@DATASETS.register_module()
class UDADataset(CocoDataset):
    CLASSES = ('triangle offset' ,
        'leftover material',
        'left and right side material',
        'open on both sides',
        'big hole opening',
        'white border')
        
    def __init__(self, source, target):
        self.source = build_dataset(source)
        self.target = build_dataset(target)
        # assert hasattr(self.source, 'flag')
        # assert hasattr(self.target, 'flag')
        print(self.source.flag)
        print(self.target.flag)
        self.flag = self.target.flag
        self.CLASSES = self.target.CLASSES
        assert self.target.CLASSES == self.source.CLASSES
        # self.PALETTE = self.target.PALETTE
        # assert target.PALETTE == source.PALETTE
        self.source_len = len(self.source) 
        self.target_len = len(self.target) 
    def __len__(self):
        # return (len(self.target))
        return (max(self.source_len, self.target_len))
    
    def __getitem__(self, idx):
        if idx < self.source_len:
            s1 = self.source[idx]
        else:
            i1 = np.random.choice(range(self.source_len))
            s1 = self.source[i1]
        if idx < self.target_len:
            t1 = self.target[idx]
        else:
            i1 = np.random.choice(range(self.target_len))
            t1 = self.target[i1]
        return {
            **s1, 'target_img_metas': t1['img_metas'],
            'target_img': t1['img']
        }