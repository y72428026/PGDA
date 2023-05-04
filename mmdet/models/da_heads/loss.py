"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F

# from .consistency_loss import consistency_loss
# from maskrcnn_benchmark.modeling.poolers import Pooler

class DALossComputation(object):
    """
    This class computes the DA loss.
    """
        
    # def prepare_masks(self, targets):
    #     masks = []
    #     for targets_per_image in targets:
    #         is_source = targets_per_image.get_field('is_source')
    #         mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
    #         masks.append(mask_per_image)
    #     return masks

    def __call__(self, da_img_per_level, is_source):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        # for da_img_per_level in da_img:
        N, A, H, W = da_img_per_level.shape
        da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
        da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32) * is_source + \
            torch.ones_like(da_img_per_level, dtype=torch.float32) * (1-is_source)
        da_img_per_level = da_img_per_level.reshape(N, -1)
        da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
        # da_img_loss = F.binary_cross_entropy_with_logits(
        #     da_img_per_level, da_img_label_per_level
        # )
        da_img_loss = torch.nn.functional.mse_loss(da_img_per_level, da_img_label_per_level)
        # a = torch.zeros([2,10])
        # l = torch.ones([2,10])
        # print(F.binary_cross_entropy_with_logits( a, l )) # 0.69 = log_e(0.5)
        # l = torch.zeros([2,10])
        # print(F.binary_cross_entropy_with_logits( a, l )) # 0.69 = log_e(0.5)
        # input('da loss')

        return da_img_loss 
'''
def __call__(self, da_img, da_img_consist, is_source):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        # masks = self.prepare_masks(targets)
        # masks = torch.cat(masks, dim=0)
        # masks = is_source
        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            # da_img_label_per_level[masks, :] = 1
            da_img_label_per_level[...] = is_source

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
            
        da_img_flattened = torch.cat(da_img_flattened, dim=0)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
        
        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )
        # da_ins_loss = F.binary_cross_entropy_with_logits(
        #     torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        # )

        # da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)
        # da_consist_loss = 0
        return da_img_loss 
'''

# def make_da_heads_loss_evaluator():
#     loss_evaluator = DALossComputation()
#     return loss_evaluator
