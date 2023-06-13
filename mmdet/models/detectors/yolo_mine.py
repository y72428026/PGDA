# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
import torch

from ..builder import DETECTORS
from .yolo import YOLOV3


@DETECTORS.register_module()
class YOLOV3_mine(YOLOV3):
    def extract_feat_mine(self, img):
        """Directly extract features from the backbone+neck."""
        x1 = self.backbone(img)
        if self.with_neck:
            x2 = self.neck(x1)
        return x1, x2
    