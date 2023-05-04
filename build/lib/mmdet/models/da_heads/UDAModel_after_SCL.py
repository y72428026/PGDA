from mmdet.models import DETECTORS, build_head 
# from mmdet.models.detectors import SingleStageDetector
from mmdet.models.da_heads.grl import GradientScalarLayer 
from mmdet.utils import add_prefix
from mmcv.parallel import MMDistributedDataParallel
from copy import deepcopy
import torch, math, torch.nn as nn
from mmdet.models.da_heads.UDAModel import UDAModel
# def get_module(module):
#     """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

#     Args:
#         module (MMDistributedDataParallel | nn.ModuleDict): The input
#             module that needs processing.

#     Returns:
#         nn.ModuleDict: The ModuleDict of multiple networks.
#     """
#     if isinstance(module, MMDistributedDataParallel):
#         return module.module

#     return module

# def _params_equal(ema_model, model):
#     for ema_param, param in zip(ema_model.named_parameters(),
#                                 model.named_parameters()):
#         if not torch.equal(ema_param[1].data, param[1].data):
#             # print("Difference in", ema_param[0])
#             return False
#     return True

@DETECTORS.register_module()
class UDAModel_SCL(UDAModel):
    def __init__(self, model_net, 
                    da_head=None, 
                    da_ano_head=None, 
                    da_pred_head=None, 
                    enable_category_loss=False,
                    category_weight=0,
                    enable_ease_loss=False,
                    ease_weight=0,
                    auxiliary_head_num=0, 
                    **cfg) -> None:
        super(UDAModel_SCL, self).__init__(
            model_net, 
            da_head, 
            da_ano_head, 
            da_pred_head, 
            enable_category_loss,
            category_weight,
            enable_ease_loss,
            ease_weight,
            auxiliary_head_num, 
            **cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      target_img=None,
                      target_img_metas=None,
                      gt_bboxes_ignore=None):
        losses = dict()
        model = self.get_model()
        self.local_iter += 1
        # feat.shape 
        '''
        # [   torch.Size([8, 512, 19, 18]), 
        #     torch.Size([8, 256, 38, 36]), 
        #     torch.Size([8, 128, 76, 72])]
        '''
        # trg
        feat = model.extract_feat(target_img)
        trg_da_loss = add_prefix(self.da_head(feat[-3:], is_source=False),'trg')
        losses.update(trg_da_loss)
        feat_maps, pred_maps = model.bbox_head.return_feat_pred(feat)
        trg_da_ano_loss = add_prefix(self.da_ano_head(feat_maps[-3:], is_source=False), 'trg_ano')
        trg_da_pred_loss = add_prefix(self.da_pred_head(pred_maps[-3:], is_source=False), 'trg_pred')
        losses.update(trg_da_ano_loss)
        losses.update(trg_da_pred_loss)
        # aux cls loc loss
        if self.auxiliary_head_num != 0:
            feat_grl = [self.grl_img(feat_per_layer) for feat_per_layer in feat]
            losses.update(add_prefix(self.cal_aux_cls_loc_loss(feat_grl, is_source=False), 'trg'))
        # cate loss
        if self.enable_category_loss or self.enable_ease_loss:
            trg_anchor, trg_weight = self.cal_anchor_and_weight(
                feat_maps=feat_maps, pred_maps=pred_maps, sign='trg')
        
        # src
        feat = model.extract_feat(img)
        src_da_loss = add_prefix(self.da_head(feat[-3:], is_source=True),'src')
        losses.update(src_da_loss)
        feat_maps, pred_maps = model.bbox_head.return_feat_pred(feat)
        src_da_ano_loss = add_prefix(self.da_ano_head(feat_maps[-3:], is_source=True), 'src_ano')
        src_da_pred_loss = add_prefix(self.da_pred_head(pred_maps[-3:], is_source=True), 'src_pred')
        losses.update(src_da_ano_loss)
        losses.update(src_da_pred_loss)
        # auxhead cls loc loss
        if self.auxiliary_head_num != 0:
            feat_grl = [self.grl_img(feat_per_layer) for feat_per_layer in feat]
            losses.update(add_prefix(self.cal_aux_cls_loc_loss(feat_grl, is_source=True), 'src'))
        # cate loss
        if self.enable_category_loss or self.enable_ease_loss:
            src_anchor, src_weight = self.cal_anchor_and_weight(
                feat_maps=feat_maps, pred_maps=pred_maps, sign='src')
            
        # head loss
        loss = model.bbox_head.forward_train(
            feat, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(loss) 

        # auxhead loss, ban bp
        if self.auxiliary_head_num != 0:
            feat_clone = [ele.clone().detach() for ele in feat]
            for i in range(self.auxiliary_head_num):
                loss = self.auxiliary_head[i].forward_train(
                feat_clone, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
                losses.update(add_prefix(loss, f'aux_{i}')) 
        
        # category loss
        if self.enable_category_loss:
            for i in range(3):
                loss = self.category_loss(src_anchor[i], src_weight[i], trg_anchor[i], trg_weight[i], layer_lvl=i)
                losses.update(add_prefix(loss, f'anchor_{i}'))
        
        # ease loss
        if self.enable_ease_loss:
            for i in range(3):
                loss = self.ease_loss(src_anchor[i], src_weight[i], trg_anchor[i], trg_weight[i], layer_lvl=i)
                losses.update(add_prefix(loss, f'anchor_{i}'))
        # torch.autograd.set_detect_anomaly(True) 
        return losses


"""
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      target_img=None,
                      target_img_metas=None,
                      gt_bboxes_ignore=None):
        losses = dict()
        model = self.get_model()
        # feat.shape 
        '''
        # [   torch.Size([8, 512, 19, 18]), 
        #     torch.Size([8, 256, 38, 36]), 
        #     torch.Size([8, 128, 76, 72])]
        '''
        # trg
        feat = model.extract_feat(target_img)
        trg_da_loss = add_prefix(self.da_head(feat[-3:], is_source=False),'trg')
        losses.update(trg_da_loss)
        feat_maps, pred_maps = model.bbox_head.return_feat_pred(feat)
        trg_da_ano_loss = add_prefix(self.da_ano_head(feat_maps[-3:], is_source=False), 'trg_ano')
        trg_da_pred_loss = add_prefix(self.da_pred_head(pred_maps[-3:], is_source=False), 'trg_pred')
        losses.update(trg_da_ano_loss)
        losses.update(trg_da_pred_loss)
        if self.enable_category_loss:
            trg_anchor, trg_weight = self.cal_anchor_and_weight(
                feat_maps=feat_maps, pred_maps=pred_maps )
        # aux cls loc loss
        if self.auxiliary_head_num != 0:
            feat_grl = [self.grl_img(feat_per_layer) for feat_per_layer in feat]
            losses.update(add_prefix(self.cal_aux_cls_loc_loss(feat_grl, is_source=False), 'trg'))
        
        # src
        feat = model.extract_feat(img)
        src_da_loss = add_prefix(self.da_head(feat[-3:], is_source=True),'src')
        losses.update(src_da_loss)
        feat_maps, pred_maps = model.bbox_head.return_feat_pred(feat)
        src_da_ano_loss = add_prefix(self.da_ano_head(feat_maps[-3:], is_source=True), 'src_ano')
        src_da_pred_loss = add_prefix(self.da_pred_head(pred_maps[-3:], is_source=True), 'src_pred')
        losses.update(src_da_ano_loss)
        losses.update(src_da_pred_loss)
        # aux cls loc loss
        if self.auxiliary_head_num != 0:
            feat_grl = [self.grl_img(feat_per_layer) for feat_per_layer in feat]
            losses.update(add_prefix(self.cal_aux_cls_loc_loss(feat_grl, is_source=True), 'src'))

        if self.enable_category_loss:
            src_anchor, src_weight = self.cal_anchor_and_weight(
                feat_maps=feat_maps, pred_maps=pred_maps)
            
        # head loss
        loss = model.bbox_head.forward_train(
            feat, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(loss) 

        # aux head loss, ban bp
        if self.auxiliary_head_num != 0:
            feat_clone = [ele.clone().detach() for ele in feat]
            for i in range(self.auxiliary_head_num):
                loss = self.auxiliary_head[i].forward_train(
                feat_clone, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
                losses.update(add_prefix(loss, f'aux_{i}')) 
        
        # category loss
        if self.enable_category_loss:
            for i in range(3):
                loss = self.category_loss(src_anchor[i], src_weight[i], trg_anchor[i], trg_weight[i], lossweight=1)
                losses.update(add_prefix(loss, f'anchor_{i}'))
        # torch.autograd.set_detect_anomaly(True) 
        return losses
"""
"""  
    # aux head pred loss
    def cal_aux_cls_loc_loss(self, feat, is_source=True):
        assert self.auxiliary_head_num != 0
        # feat.shape 
        '''
        # [   torch.Size([8, 512, 15, 19]), 
        #     torch.Size([8, 256, 30, 38]), 
        #     torch.Size([8, 128, 60, 76])]
        '''
        eps = torch.finfo(torch.float32).eps
        pred_maps_list = [[],[],[]]
        losses = dict()
        sign = is_source * -1 + (1 - is_source) * 1
        for i in range(self.auxiliary_head_num):
            _, pred_maps = self.auxiliary_head[i].forward_da(feat)
            for j in range(3): # 3 anchor
                pred_maps_list[j].append(pred_maps[j]) # [n,-1,11]
        for j in range(3): # 3 anchor
            pred_maps_stack = torch.stack(pred_maps_list[j], dim=1) # torch.Size([8, aux_num, 3hw, 11])
            pred_maps_loc = pred_maps_stack[...,:4] # torch.Size([8, aux_num, 3hw, 4])
            pred_maps_loc[...,:2] = pred_maps_loc[...,:2].sigmoid()
            pred_maps_conf = (pred_maps_stack[...,4]).sigmoid() # torch.Size([8, aux_num, 3hw])
            pred_maps_cls = pred_maps_stack[...,5:].sigmoid() # torch.Size([8, aux_num, 3hw, cls_num])
            pred_maps_cls_mean = pred_maps_cls / (pred_maps_cls.sum(dim=-1, keepdims=True)) # torch.Size([8, aux_num, 3hw, cls_num])
            # cls 
            pred_entropy = torch.sum(-pred_maps_cls_mean * torch.log(pred_maps_cls_mean), dim=1) # torch.Size([8, 3hw, cls_num])
            pred_mean = pred_maps_cls_mean.mean(dim=1) # torch.Size([8, 3hw, cls_num])
            loss_cls = pred_entropy * pred_mean
            loss_cls = loss_cls.sum(dim=-1) # torch.Size([8, 3hw])
            # loc
            pred_maps_loc_mean = pred_maps_loc.mean(dim=1, keepdims=True) # torch.Size([8, 1, 3hw, 4])
            loss_loc = torch.norm(pred_maps_loc - pred_maps_loc_mean, dim=-1) # torch.Size([8, aux_num, 3hw])
            loss_loc = (loss_loc/4/math.sqrt(self.auxiliary_head_num)).sum(dim=1) # torch.Size([8, 3hw])
            # mask and loss 
            pred_mask = pred_maps_conf.mean(dim=1) > 0.5   # torch.Size([8, 3hw])
            losses['loss_cls'] = sign * self.cls_weight * (loss_cls * pred_mask).sum() / (pred_mask.sum() + eps )
            losses['loss_loc'] = sign * self.loc_weight * (loss_loc * pred_mask).sum() / (pred_mask.sum() + eps )
            losses['pred_conf_mean'] = pred_mask.float().mean()
        return losses 
    
    def simple_test(self, img, img_metas, rescale=False):
        return self.get_model().simple_test(img, img_metas, rescale)
    
    def aug_test(self, imgs, img_metas, rescale=False):
        return self.get_model().aug_test(imgs, img_metas, rescale)

    def cal_anchor_and_weight(self, feat_maps, pred_maps):
        # feat.shape 
        '''
        # [   torch.Size([8, 512, 19, 18]), 
        #     torch.Size([8, 256, 38, 36]), 
        #     torch.Size([8, 128, 76, 72])]
        '''
        # pred.shape   [(N,33,H,W)*3]
        # feat_maps, pred_maps = self.model.bbox_head.return_feat_pred(feat) 
        anchor_list = []
        weight_list = []
        eps = torch.finfo(torch.float).eps
        for i in range(3):
            N, C, H, W = feat_maps[i].shape
            feat_map_per_level = feat_maps[i].permute(0, 2, 3, 1) # (N, H, W, C)
            pred_map_per_level = pred_maps[i].permute(0, 2, 3, 1).reshape(N, H, W, 3, 11)
            pred_map_per_level_conf = (pred_map_per_level[...,4]).sigmoid() # (N, H, W, 3)
            pred_map_per_level_conf = pred_map_per_level_conf.mean(dim=-1) # (N, H, W)
            pred_map_per_level_cls = pred_map_per_level[...,5:].sigmoid() # (N, H, W, 3, 6) 
            pred_map_per_level_cls = pred_map_per_level_cls.mean(dim=-2) # (N, H, W, 6)
            anchor = []
            weight = torch.zeros([6,1], device=pred_map_per_level_cls.device, dtype=torch.float)
            for j in range(6):
                anchor.append((feat_map_per_level * pred_map_per_level_cls[...,j].unsqueeze(-1)).sum(dim=[0,1,2]) / (
                    pred_map_per_level_cls.sum() + eps ) )
                weight[j] = pred_map_per_level_cls[...,j].max() 
                weight[j] = (weight[j] > 0.5) * (1 - weight[j])**2 
            anchor = torch.stack(anchor, dim=0)
            anchor_list.append(anchor) 
            weight_list.append(weight)
        return anchor_list, weight_list

    def category_loss(self, src_anchor, src_weight, trg_anchor, trg_weight, lossweight=1):
        K, C = src_anchor.shape
        eps = torch.finfo(torch.float).eps
        dist = torch.zeros([K,K], dtype=torch.float, device=src_anchor.device)
        for i in range(K):
            for j in range(K):
                dist[i,j] = torch.norm(src_anchor - trg_anchor)
        weight = src_weight.transpose(0,1) @ trg_weight + eps
        loss_intra = (src_weight.transpose(0,1) @ dist @ trg_weight) / weight * lossweight
        loss_inter = self.inter_loss(src_anchor, src_weight, trg_anchor, trg_weight) 
        loss_inter += self.inter_loss(src_anchor, src_weight, src_anchor, src_weight) 
        loss_inter += self.inter_loss(trg_anchor, trg_weight, trg_anchor, trg_weight) 
        loss_inter = loss_inter / 3 * lossweight
        losses = dict()
        losses['loss_intra'] = loss_intra
        losses['loss_inter'] = loss_inter
        return losses
   
    def inter_loss(self, src_anchor, src_weight, trg_anchor, trg_weight):
        K, C = src_anchor.shape
        eps = torch.finfo(torch.float).eps
        dist = torch.zeros([K,K], dtype=torch.float, device=src_anchor.device)
        for i in range(K):
            for j in range(K):
                a = 1 - torch.norm(src_anchor - trg_anchor)
                dist[i,j] = 0.5 * (a + torch.abs(a)) 
        weight = src_weight.transpose(0,1) @ trg_weight
        loss = (src_weight.transpose(0,1) @ dist @ trg_weight) / (weight + eps)
        return loss
        
    # train_step for lossW
    '''
    def train_step(self, data, optimizer):
        The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        
        losses = self(**data)
        if self.loss_keys == None:
            self.loss_keys = [_key for _key, _value in losses.items()
                if 'loss' in _key]
        for i, key in enumerate(self.loss_keys):
            weight = torch.exp(-self.loss_weight[i])
            if isinstance(losses[key], list):
                for i in range(len(losses[key])): 
                    losses[key][i] *= weight
            else:
                losses[key] *= weight
            losses[f'{key}_w'] = self.loss_weight[i]
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs
    '''
"""