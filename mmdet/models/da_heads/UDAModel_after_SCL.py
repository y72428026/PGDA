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
                    alpha_mb=1,
                    cfa_v=9,
                    auxiliary_head_num=0, 
                    **cfg) -> None:
        super(UDAModel_SCL, self).__init__(
            model_net, 
            da_head=da_head, 
            da_ano_head=da_ano_head, 
            da_pred_head=da_pred_head, 
            enable_category_loss=enable_category_loss,
            category_weight=category_weight,
            enable_ease_loss=enable_ease_loss,
            ease_weight=ease_weight,
            alpha_mb=alpha_mb,
            cfa_v=cfa_v,
            auxiliary_head_num=auxiliary_head_num, 
            **cfg)

    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   target_img=None,
    #                   target_img_metas=None,
    #                   gt_bboxes_ignore=None):
    #     losses = dict()
    #     model = self.get_model()
    #     self.local_iter += 1
    #     # feat.shape 
    #     '''
    #     # [   torch.Size([8, 512, 19, 18]), 
    #     #     torch.Size([8, 256, 38, 36]), 
    #     #     torch.Size([8, 128, 76, 72])]
    #     '''
    #     # trg
    #     feat = model.extract_feat(target_img)
    #     trg_da_loss = add_prefix(self.da_head(feat[-3:], is_source=False),'trg')
    #     losses.update(trg_da_loss)
    #     feat_maps, pred_maps = model.bbox_head.return_feat_pred(feat)
    #     trg_da_ano_loss = add_prefix(self.da_ano_head(feat_maps, is_source=False), 'trg_ano')
    #     # trg_da_pred_loss = add_prefix(self.da_pred_head(pred_maps, is_source=False), 'trg_pred')
    #     losses.update(trg_da_ano_loss)
    #     # losses.update(trg_da_pred_loss)
    #     # aux cls loc loss
    #     if self.auxiliary_head_num != 0:
    #         feat_grl = [self.grl_img(feat_per_layer) for feat_per_layer in feat]
    #         losses.update(add_prefix(self.cal_aux_cls_loc_loss(feat_grl, is_source=False), 'trg'))
    #     # cate loss
    #     if self.enable_category_loss or self.enable_ease_loss:
    #         # trg_anchor, trg_weight = self.cal_anchor_and_weight(
    #         #     feat_maps=feat_maps, pred_maps=pred_maps, sign='trg')
    #         trg_anchor, trg_weight = self.cal_anchor_and_weight(
    #             feat_maps=feat[-3:], pred_maps=pred_maps, sign='trg')
    #     # src
    #     feat = model.extract_feat(img)
    #     src_da_loss = add_prefix(self.da_head(feat[-3:], is_source=True),'src')
    #     losses.update(src_da_loss)
    #     feat_maps, pred_maps = model.bbox_head.return_feat_pred(feat)
    #     # input([ele.shape for ele in feat])
    #     # input([ele.shape for ele in feat_maps])
    #     src_da_ano_loss = add_prefix(self.da_ano_head(feat_maps, is_source=True), 'src_ano')
    #     # src_da_pred_loss = add_prefix(self.da_pred_head(pred_maps, is_source=True), 'src_pred')
    #     losses.update(src_da_ano_loss)
    #     # losses.update(src_da_pred_loss)
    #     # auxhead cls loc loss
    #     if self.auxiliary_head_num != 0:
    #         feat_grl = [self.grl_img(feat_per_layer) for feat_per_layer in feat]
    #         losses.update(add_prefix(self.cal_aux_cls_loc_loss(feat_grl, is_source=True), 'src'))
    #     # cate loss
    #     if self.enable_category_loss or self.enable_ease_loss:
    #         # src_anchor, src_weight = self.cal_anchor_and_weight(
    #         #     feat_maps=feat_maps, pred_maps=model.bbox_head.return_target_maps_list(
    #         #     pred_maps, gt_bboxes, gt_labels, img_metas), sign='src')
    #         src_anchor, src_weight = self.cal_anchor_and_weight(
    #             feat_maps=feat[-3:], pred_maps=model.bbox_head.return_target_maps_list(
    #             pred_maps, gt_bboxes, gt_labels, img_metas), sign='src')
            
    #     # head loss
    #     loss = model.bbox_head.forward_train(
    #         feat, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
    #     losses.update(loss) 

    #     # auxhead loss, ban bp
    #     if self.auxiliary_head_num != 0:
    #         feat_clone = [ele.clone().detach() for ele in feat]
    #         for i in range(self.auxiliary_head_num):
    #             loss = self.auxiliary_head[i].forward_train(
    #             feat_clone, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
    #             losses.update(add_prefix(loss, f'aux_{i}')) 
        
    #     # category loss
    #     if self.enable_category_loss:
    #         for i in range(3):
    #             loss = self.category_loss(src_anchor[i], src_weight[i], trg_anchor[i], trg_weight[i], layer_lvl=i)
    #             losses.update(add_prefix(loss, f'anchor_{i}'))
        
    #     # ease loss
    #     if self.enable_ease_loss:
    #         for i in range(3):
    #             loss = self.ease_loss(src_anchor[i], src_weight[i], trg_anchor[i], trg_weight[i], layer_lvl=i)
    #             losses.update(add_prefix(loss, f'anchor_{i}'))
        
    #     ### !!!
    #     # update memory bank
    #     if self.enable_category_loss or self.enable_ease_loss:
    #         for i in range(len(src_anchor)):
    #             self.update_memory_bank(src_anchor[i], src_weight[i], i, 'src')
    #             self.update_memory_bank(trg_anchor[i], trg_weight[i], i, 'trg')
    #     return losses


if __name__ == '__main__':
    dev = "cuda"
    torch.cuda.set_device(7)
    from mmcv.utils import Config
    from mmdet.models import build_detector, build_detector_mine
    import os
    cfg = Config.fromfile('/home/yebh/mmdetection/configs/uda_608/uda_yolov3_T2L_SCL_75k75k0-00625-a07.py')
    cfg.work_dir='work_dirs/test'
    udamodel = build_detector_mine(
        cfg,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()

    src_image = torch.randint(255, [1,3,640,640]).float().cuda()
    trg_image = torch.randint(255, [1,3,640,640]).float().cuda()

    src_feat = udamodel.model.extract_feat(src_image)
    print([ ele.shape for ele in src_feat]) # [torch.Size([1, 512, 20, 20]), torch.Size([1, 256, 40, 40]), torch.Size([1, 128, 80, 80])]
    feat_maps, pred_maps = udamodel.model.bbox_head.return_feat_pred(src_feat)
    print([ ele.shape for ele in feat_maps]) # [torch.Size([1, 1024, 20, 20]), torch.Size([1, 512, 40, 40]), torch.Size([1, 256, 80, 80])]
    print([ ele.shape for ele in pred_maps]) # [torch.Size([1, 33, 20, 20]), torch.Size([1, 33, 40, 40]), torch.Size([1, 33, 80, 80])]
    trg_anchor, trg_weight = udamodel.cal_anchor_and_weight(feat_maps=feat_maps, pred_maps=pred_maps, sign='trg')
    print([ ele.shape for ele in trg_anchor])# [torch.Size([6, 512]), torch.Size([6, 256]), torch.Size([6, 128])]
    print([ ele.shape for ele in trg_weight])# [torch.Size([6, 1]), torch.Size([6, 1]), torch.Size([6, 1])]
