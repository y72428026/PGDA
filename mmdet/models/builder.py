# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
UDA = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

import copy
def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build_detector_mine(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    
    # # input(cfg.keys())
    # if 'uda' in cfg:
    #     # from mmdet.models.detectors.UDAModel import UDAModel
    #     from mmdet.models.da_heads.UDAModel_after_SCL import UDAModel_SCL
    #     from mmdet.models.da_heads.UDAModel import UDAModel
    #     model = copy.deepcopy(cfg.model)
    #     model_net = build_detector_mine(model, train_cfg=train_cfg, test_cfg=test_cfg)
    #     cfg.uda['model'] = model
    #     cfg.uda['max_epochs'] = cfg.runner.max_epochs 
    #     cfg.uda['work_dir'] = cfg.get('work_dir','')
    #     if cfg.uda.type == "UDAModel":
    #         return UDAModel(model_net, **cfg.uda)
    #     elif cfg.uda.type == "UDAModel_SCL":
    #         return UDAModel_SCL(model_net, **cfg.uda)
    # elif 'model' in cfg:
    #     return DETECTORS.build(
    #         cfg.model, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    # else:
    #     return DETECTORS.build(
    #         cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    if 'uda' in cfg:
        # from mmdet.models.detectors.UDAModel import UDAModel
        from mmdet.models.da_heads.UDAModel_after_SCL import UDAModel_SCL
        from mmdet.models.da_heads.UDAModel import UDAModel
        from mmdet.models.da_heads import UDAModel_Xtt
        from mmdet.models.da_heads import UDAModel_Xts
        from mmdet.models.da_heads import UDAModel_Xss
        from mmdet.models.da_heads import UDAModel_Xst
        from mmdet.models.da_heads import UDAModel_Ost
        model = copy.deepcopy(cfg.model)
        model_net = build_detector_mine(model, train_cfg=train_cfg, test_cfg=test_cfg)
        cfg.uda['model'] = model
        cfg.uda['max_epochs'] = cfg.runner.max_epochs 
        cfg.uda['work_dir'] = cfg.get('work_dir','')
        if cfg.uda.type == "UDAModel":
            return UDAModel(model_net, **cfg.uda)
        elif cfg.uda.type == "UDAModel_SCL":
            print('SCL')
            return UDAModel_SCL(model_net, **cfg.uda)
        elif cfg.uda.type == "UDAModel_Xtt":
            print('Xtt')
            return UDAModel_Xtt(model_net, **cfg.uda)
        elif cfg.uda.type == "UDAModel_Xst":
            print('Xst')
            return UDAModel_Xst(model_net, **cfg.uda)
        elif cfg.uda.type == "UDAModel_Xss":
            print('Xss')
            return UDAModel_Xss(model_net, **cfg.uda)
        elif cfg.uda.type == "UDAModel_Xts":
            print('Xts')
            return UDAModel_Xts(model_net, **cfg.uda)
        elif cfg.uda.type == "UDAModel_Ost":
            print('Ost')
            return UDAModel_Ost(model_net, **cfg.uda)
        else:
            raise NotImplementedError(f"UDA type {cfg.uda.type} not implemented")
    elif 'model' in cfg:
        return DETECTORS.build(
            cfg.model, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    else:
        return DETECTORS.build(
            cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
