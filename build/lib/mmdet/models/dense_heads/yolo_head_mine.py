# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm,
                      normal_init)
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

from .yolo_head import YOLOV3Head
@HEADS.register_module()
class YOLOV3Head_mine(YOLOV3Head):

    def __init__(self,
                 num_classes,
                 in_channels,
                 in_channels_mine=[64,128],
                 **kwargs):
        self.in_channels_mine = in_channels_mine
        super(YOLOV3Head_mine, self).__init__(num_classes, in_channels, **kwargs)
        self._init_layers_mine()
        # TODO just avgpool or avgpool then repeat
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _init_layers_mine(self):
        # in_channels=[64, 128],
        # out_channels=[128, 128],
        # conv + bn + LeakyReLU(default)
        conv_11 = ConvModule(
            self.in_channels_mine[0],
            32,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # conv_12 = ConvModule(
        #     32,
        #     32,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        
        conv_21 = ConvModule(
            self.in_channels_mine[1],
            64,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # conv_22 = ConvModule(
        #     64,
        #     32,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        #####
        # conv_l3 = nn.Conv2d(32, 64, 1)
        # self.convs4Lfeat_1 = nn.Sequential(conv_11, conv_12, conv_l3)
        # self.convs4Lfeat_2 = nn.Sequential(conv_21, conv_22, conv_l3)
        #####
        conv_l3 = nn.Conv2d(32, 64, 1)
        conv_23 = nn.Conv2d(64, 64, 1)
        self.convs4Lfeat_1 = nn.Sequential(conv_11, conv_l3)
        self.convs4Lfeat_2 = nn.Sequential(conv_21, conv_23)

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        # in_channels=[512, 256, 128],
        # out_channels=[1024, 512, 256],
        for i in range(self.num_levels):
            # conv + bn + LeakyReLU(default)
            '''
            128 for feats of layer 0 and 1 outputed by backbone
            '''
            conv_bridge = ConvModule(
                self.in_channels[i]+128,
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_base_priors * self.num_attrib, 1)
            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def return_feat_pred(self, feats_maps):
        return self.return_feat_pred_avg_cat(feats_maps=feats_maps)
        # return self.return_feat_pred_avg_repeat_cat(feats_maps)
    # not use
    def return_feat_pred_avg_repeat_cat(self, feats_maps):
        assert len(feats_maps) >= self.num_levels
        feats_Llvl = feats_maps[:-self.num_levels]
        # # detach to ban bp
        # feats_Llvl = [feat.detach().clone() for feat in feats_Llvl]
        feats = feats_maps[-self.num_levels:]
        feat_maps = []
        pred_maps = []

        x1 = self.convs4Lfeat_1(feats_Llvl[0])
        x1 = self.avgpool(x1)
        x2 = self.convs4Lfeat_2(feats_Llvl[1])
        x2 = self.avgpool(x2)
        x_cat = torch.cat((x1,x2), dim=1)
        for i in range(self.num_levels):
            x = feats[i]
            _,_,H,W = x.shape
            x = self.convs_bridge[i](x)
            x_cat_repeat = x_cat.repeat(1,1,H,W)
            x = torch.cat((x, x_cat_repeat), dim=1)
            feat_maps.append(x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        return feat_maps, pred_maps
    # not use
    def return_feat_pred_avg_cat_backup(self, feats_maps):
        assert len(feats_maps) >= self.num_levels
        feats_Llvl = feats_maps[:-self.num_levels]
        # detach to ban bp
        # feats_Llvl = [feat.detach().clone() for feat in feats_Llvl]
        feats = feats_maps[-self.num_levels:]
        feat_maps = []
        pred_maps = []

        x1 = self.convs4Lfeat_1(feats_Llvl[0])
        x2 = self.convs4Lfeat_2(feats_Llvl[1])
        for i in range(self.num_levels):
            x = feats[i]
            _,_,H,W = x.shape
            x = self.convs_bridge[i](x)
            x1_avg = torch.nn.functional.adaptive_avg_pool2d(x1, [H,W])
            x2_avg = torch.nn.functional.adaptive_avg_pool2d(x2, [H,W])
            feat_maps.append(x)
            x = torch.cat((x, x1_avg, x2_avg), dim=1)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        return feat_maps, pred_maps

    def return_feat_pred_avg_cat(self, feats_maps):
        assert len(feats_maps) >= self.num_levels
        feats_Llvl = feats_maps[:-self.num_levels]
        # detach to ban bp
        # feats_Llvl = [feat.detach().clone() for feat in feats_Llvl]
        feats = feats_maps[-self.num_levels:]
        feat_maps = []
        pred_maps = []

        x1 = self.convs4Lfeat_1(feats_Llvl[0])
        x2 = self.convs4Lfeat_2(feats_Llvl[1])

        for i in range(self.num_levels):
            x = feats[i]
            _,_,H,W = x.shape
            x1_avg = torch.nn.functional.adaptive_avg_pool2d(x1, [H,W])
            x2_avg = torch.nn.functional.adaptive_avg_pool2d(x2, [H,W])
            x = torch.cat((x, x1_avg, x2_avg), dim=1)
            x = self.convs_bridge[i](x)
            feat_maps.append(x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        return feat_maps, pred_maps


    def forward(self, feat):
        _, pred_maps = self.return_feat_pred(feat)
        return tuple(pred_maps),
    
    def forward_da(self, feats):
        N = feats[0].shape[0]
        feat_maps, pred_maps = self.return_feat_pred(feats)
        pred_maps = [pred_map.permute(0, 2, 3,
                                        1).reshape(N, -1,
                                                   self.num_attrib) for pred_map in pred_maps ]
        return feat_maps, tuple(pred_maps)

    # def loss_single(self, pred_map, target_map, neg_map):
    #     """Compute loss of a single image from a batch.

    #     Args:
    #         pred_map (Tensor): Raw predictions for a single level.
    #         target_map (Tensor): The Ground-Truth target for a single level.
    #         neg_map (Tensor): The negative masks for a single level.

    #     Returns:
    #         tuple:
    #             loss_cls (Tensor): Classification loss.
    #             loss_conf (Tensor): Confidence loss.
    #             loss_xy (Tensor): Regression loss of x, y coordinate.
    #             loss_wh (Tensor): Regression loss of w, h coordinate.
    #     """
    #     # n,c,h,w -> n,h,w,c -> n,h*w*3, 5+classnum 
    #     num_imgs = len(pred_map) # 
    #     pred_map = pred_map.permute(0, 2, 3,
    #                                 1).reshape(num_imgs, -1, self.num_attrib)
    #     neg_mask = neg_map.float()
    #     pos_mask = target_map[..., 4]
    #     pos_and_neg_mask = neg_mask + pos_mask
    #     pos_mask = pos_mask.unsqueeze(dim=-1)
    #     if torch.max(pos_and_neg_mask) > 1.:
    #         warnings.warn('There is overlap between pos and neg sample.')
    #         pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

    #     pred_xy = pred_map[..., :2]
    #     pred_wh = pred_map[..., 2:4]
    #     pred_conf = pred_map[..., 4]
    #     pred_label = pred_map[..., 5:]

    #     target_xy = target_map[..., :2]
    #     target_wh = target_map[..., 2:4]
    #     target_conf = target_map[..., 4]
    #     target_label = target_map[..., 5:]
    #     # print('new') 
    #     # print(pos_mask.shape)   # [8, 798, 1]
    #     # print(pos_and_neg_mask.shape)   #  [8, 798]
    #     # print(pred_conf.shape)  # [8, 798]
    #     # print(pred_label.shape) # [8, 798, 6]
    #     # input('check tensor shape')
    #     ### mine code
    #     pred_label_max = torch.sigmoid(pred_label.max(dim=-1, keepdim=True)[0])
    #     pred_label_mask = (1 - pred_label_max) * pos_mask 
    #     # print(pred_label_mask.shape) # torch.Size([8, 798, 1])
    #     # loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
    #     loss_cls = self.loss_cls(pred_label, target_label, weight=pred_label_mask)

    #     # conf_diff = torch.sigmoid(pred_conf - target_conf)
    #     # conf_mask = pos_and_neg_mask * 2 * torch.abs(conf_diff - 0.5)
    #     # print(conf_mask.shape) # torch.Size([8, 798])
    #     loss_conf = self.loss_conf(
    #         # pred_conf, target_conf, weight=conf_mask)
    #         pred_conf, target_conf, weight=pos_and_neg_mask)

    #     loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
    #     loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)
    #     # print(loss_cls)
    #     # print(loss_cls.shape)
    #     # input("check loss shape")

    #     return loss_cls, loss_conf, loss_xy, loss_wh