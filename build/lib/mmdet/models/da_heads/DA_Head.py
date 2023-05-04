# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from .grl import GradientScalarLayer
from ..builder import HEADS 
from .loss import DALossComputation

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels): # 1024
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(in_channels//2, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        t = self.conv2_da(t)
        # print(self.conv1_da.bias.mean())
        # print(self.conv2_da.bias.mean())
        # input('-,-')
        return t


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels): # 2048
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x

@HEADS.register_module()
class DomainAdaptationHead(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, in_channels, img_weight=1, **cfg):
        super(DomainAdaptationHead, self).__init__()

        # self.cfg = cfg.clone()
        # num_ins_inputs = 2048
        # in_channels = 1024
        self.in_channels = in_channels
        # self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        # self.resnet_backbone = True 
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = img_weight  
        # self.ins_weight = 1
        # self.cst_weight = 0.1
        self.grl_img = GradientScalarLayer(-1.0*0.1)
        # self.grl_ins = GradientScalarLayer(-1.0*0.1)
        self.grl_img_consist = GradientScalarLayer(1.0*0.1)
        # self.grl_ins_consist = GradientScalarLayer(1.0*0.1)
        self.imghead = nn.ModuleList()
        for in_channels in self.in_channels:
            self.imghead.append(DAImgHead(in_channels))
        # self.inshead = DAInsHead(num_ins_inputs) # 2048
        # self.loss_evaluator = make_da_heads_loss_evaluator()
        self.loss_evaluator = DALossComputation()

    def forward(self, img_features, is_source=True):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # print(img_features[0].shape) # [torch.Size([2, 1024, 38, 50])]
        # print (len(da_ins_feature)) # [512, 2048, 7, 7]
        # if self.resnet_backbone:
        #     da_ins_feature = self.avgpool(da_ins_feature)
        # da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features] 
        # img_grl_fea = [self.grl_img_consist(fea) for fea in img_features] 
        # ins_grl_fea = self.grl_ins(da_ins_feature)
        # img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        # ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
        
        da_img_features = [self.imghead[i](img_grl_fea[i]) for i in range(len(img_features))]
        # da_ins_features = self.inshead(ins_grl_fea)
        # da_img_consist_features = [self.imghead[i](img_grl_consist_fea[i]) for i in range(len(img_grl_consist_fea))]
        # da_img_consist_features = self.imghead(img_grl_consist_fea)
        # da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        # da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        # da_ins_consist_features = da_ins_consist_features.sigmoid()
        if self.training:
            losses_total = {}
            loss_da_image = torch.tensor(0.).cuda()
            for i in range(len(da_img_features)):
                # da_img_loss = self.loss_evaluator(
                #     [da_img_features[i]], [da_img_consist_features[i]], is_source
                # )
                da_img_loss = self.loss_evaluator(
                    da_img_features[i], is_source
                )
                if self.img_weight > 0:
                    loss_da_image += self.img_weight * da_img_loss
                # if self.ins_weight > 0:
                #     losses[f"loss_da_instance_{i}"] = self.ins_weight * da_ins_loss
                # if self.cst_weight > 0:
                #     losses[f"loss_da_consistency_{i}"] = self.cst_weight * da_consistency_loss
                # losses_total.update(losses)
            # print(is_source)
            losses_total['loss_da_img'] = loss_da_image / len(da_img_features)
            # input(losses_total)
            return losses_total
        return {}

# def build_da_heads(cfg):
#     if cfg.MODEL.DOMAIN_ADAPTATION_ON:
#         return DomainAdaptationModule(cfg)
#     return []
