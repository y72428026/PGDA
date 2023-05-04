# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from .grl import GradientScalarLayer
from ..builder import HEADS 

class DAImgHead(nn.Module):
    def __init__(self, in_channels): # 1024
        super(DAImgHead, self).__init__()     
        self.conv1_da = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(in_channels//2, 1, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)
    def forward(self, x):
        t = F.relu(self.conv1_da(x))
        t = self.conv2_da(t)
        return t


@HEADS.register_module()
class DomainAdaptationHead(torch.nn.Module):

    def __init__(self, in_channels, img_weight=1, GAN_type='LSGAN', **cfg):
        super(DomainAdaptationHead, self).__init__()
        self.in_channels = in_channels
        self.img_weight = img_weight  
        self.grl_img = GradientScalarLayer(-1.0*0.1)
        self.grl_img_consist = GradientScalarLayer(1.0*0.1)
        self.imghead = nn.ModuleList()
        self.GAN_type = GAN_type
        self.k = 0.5
        for in_channels in self.in_channels:
            self.imghead.append(DAImgHead(in_channels))

    def forward(self, img_features, is_source=True):
        img_grl_fea = [self.grl_img(fea) for fea in img_features] 
        da_img_features = [self.imghead[i](img_grl_fea[i]) for i in range(len(img_features))]
        if self.training:
            if self.GAN_type == 'WGAN':  
                self.clip_parameter()
            losses_total = {}
            loss_da_image = torch.tensor(0.).cuda()
            for i in range(len(da_img_features)):
                da_img_loss = self.loss_func(da_img_features[i], is_source)
                if self.img_weight > 0:
                    loss_da_image += self.img_weight * da_img_loss
            losses_total['loss_da_img'] = loss_da_image / len(da_img_features)
            return losses_total
        return {}
    
    def gradient_penalty(self, src_feat, trg_feat):
        gamma = torch.rand(1).cuda()
        losses = {}
        loss_gp = torch.tensor(0.).cuda()
        if self.training:
            for i in range(len(src_feat)):
                feat = gamma * src_feat[i].detach().clone() + (1-gamma) * trg_feat[i].detach().clone()
                # total_feat.append(feat)
                da_img_loss = self.loss_func(feat, is_source=False)
                da_img_loss = self.img_weight * da_img_loss
                gradient_gp = torch.autograd.grad(da_img_loss, feat)
                gradient_norm = torch.norm(gradient_gp)
                loss_gp += torch.pow((gradient_norm - 1), 2)
            losses['loss_gp'] = loss_gp / len(src_feat)
        return losses
    
    def loss_func(self, da_img_per_level, is_source):
        N, A, H, W = da_img_per_level.shape
        da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
        da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32) * is_source + \
            torch.ones_like(da_img_per_level, dtype=torch.float32) * (1-is_source)
        da_img_per_level = da_img_per_level.reshape(N, -1)
        da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
        if self.GAN_type == 'LSGAN':
            da_img_loss = torch.nn.functional.mse_loss(da_img_per_level, da_img_label_per_level)
        elif self.GAN_type == 'WGAN' or 'WGAN_GP':
            da_img_loss = torch.nn.functional.smooth_l1_loss(da_img_per_level-da_img_label_per_level)
        if self.GAN_type == 'GAN':
            da_img_per_level = torch.sigmoid(da_img_loss)
            da_img_loss = torch.nn.functional.binary_cross_entropy(da_img_per_level, da_img_label_per_level)
        return da_img_loss 
    
    @torch.no_grad()
    def clip_parameter(self):
        # print([ele for ele in self.parameters()])
        for param in self.parameters():
            param.clamp_(-self.k, self.k)
        # print([ele for ele in self.parameters()])
        
# class DALossComputation(object):
#     def __init__(self, loss_type='LSGAN') -> None:
#         pass
#     def __call__(self, da_img_per_level, is_source):
#         N, A, H, W = da_img_per_level.shape
#         da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
#         da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32) * is_source + \
#             torch.ones_like(da_img_per_level, dtype=torch.float32) * (1-is_source)
#         da_img_per_level = da_img_per_level.reshape(N, -1)
#         da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
#         da_img_loss = torch.nn.functional.mse_loss(da_img_per_level, da_img_label_per_level)
#         return da_img_loss 

# m =  DomainAdaptationHead([3,2])
# m.clip_parameter()
