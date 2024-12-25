from mmdet.models import DETECTORS, build_head 
from mmdet.models.detectors import SingleStageDetector
from mmdet.models.da_heads.grl import GradientScalarLayer 
from mmdet.utils import add_prefix
from mmcv.parallel import MMDistributedDataParallel
from copy import deepcopy
import torch, math, torch.nn as nn
import mmcv
def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True

# @DETECTORS.register_module()
class UDAModel_Xts(SingleStageDetector):
    def __init__(self, model_net, 
                    da_backbone_head=None, 
                    da_neck_head=None, 
                    da_pred_head=None, 
                    da_output_head=None, 
                    enable_category_loss=False,
                    category_weight=0,
                    alpha_mb=1,
                    enable_ease_loss=False,
                    ease_weight=0,
                    auxiliary_head_num=0, 
                    num_classes=6,
                    cfa_v=9,
                    **cfg) -> None:
        super(SingleStageDetector, self).__init__()
        self.model = model_net
        self.cfa_v = cfa_v
        self.num_classes = num_classes
        self.da_backbone_head = build_head(da_backbone_head)
        self.da_neck_head = build_head(da_neck_head)
        self.da_pred_head = build_head(da_pred_head)
        self.da_output_head = build_head(da_output_head)
        # # aux loss
        # self.auxiliary_head_num = auxiliary_head_num
        # if auxiliary_head_num >0 :
        #     self.auxiliary_head = torch.nn.ModuleList()
        #     self.loc_weight = 1
        #     self.cls_weight = 1
        #     for _ in range(self.auxiliary_head_num):
        #         self.auxiliary_head.append(build_head(cfg['model']['bbox_head']))
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.conf_thres = cfg.get('cfa_conf_thres')
        self.pred_thres = cfg.get('cfa_pred_thres')
        self.grl_img = GradientScalarLayer(-1.0)
        self.loss_keys = None
        # # lossW 
        # if self.cfa_v == 16:
        #     self.loss_weight = nn.ParameterList([nn.Parameter(torch.tensor(1.0, dtype=torch.float)), # conf_mask
        #                                      nn.Parameter(torch.tensor(0.0, dtype=torch.float)), # pred_map_mask
        #                                      nn.Parameter(torch.tensor(0.0, dtype=torch.float)), # pred_map_per_level_conf
        #                                      ])
        # else:
        #     self.loss_weight = nn.ParameterList([nn.Parameter(torch.tensor(0.0, dtype=torch.float)) for _ in range(3)])
        # category loss
        self.enable_category_loss = enable_category_loss
        self.category_weight = category_weight 
        self.alpha_mb = alpha_mb
        self.src_memory_bank = list()
        self.trg_memory_bank = list()
        self.src_weight = list()
        self.trg_weight = list()
        # for c in da_neck_head['in_channels']:
        for c in da_pred_head['in_channels']:
            self.src_memory_bank.append(torch.zeros([self.num_classes, c], dtype=torch.float, requires_grad=False).cuda())
            self.src_weight.append(torch.zeros([self.num_classes, 1], dtype=torch.float, requires_grad=False).cuda())
            self.trg_memory_bank.append(torch.zeros([self.num_classes, c], dtype=torch.float, requires_grad=False).cuda())
            self.trg_weight.append(torch.zeros([self.num_classes, 1], dtype=torch.float, requires_grad=False).cuda())
        self.local_iter = 0
        self.eps = torch.finfo(torch.float).eps
        # self.linear_layer = torch.nn.Linear(1024, 512)

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas, **kwargs)

    def forward_train(self,
                      source_img,
                      source_img_metas,
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
        #     torch.Size([8, 128, 76, 72]) ]
        '''
        # trg
        feat_bb, feat_neck = model.extract_feat_mine(target_img)
        feat_pred, pred_maps = model.bbox_head.return_feat_pred(feat_neck)
        # print([ele.shape for ele in feat_bb])
        # print([ele.shape for ele in feat_neck])
        # print([ele.shape for ele in feat_pred])
        # print([ele.shape for ele in pred_maps])
        # input('cl')
        # [torch.Size([8, 256, 64, 80]), torch.Size([8, 512, 32, 40]), torch.Size([8, 1024, 16, 20])]
        # [torch.Size([8, 512, 16, 20]), torch.Size([8, 256, 32, 40]), torch.Size([8, 128, 64, 80])]
        # [torch.Size([8, 1024, 16, 20]), torch.Size([8, 512, 32, 40]), torch.Size([8, 256, 64, 80])]
        # [torch.Size([8, 24, 16, 20]), torch.Size([8, 24, 32, 40]), torch.Size([8, 24, 64, 80])]
        if self.da_backbone_head != None:
            trg_da_bb_loss = add_prefix(self.da_backbone_head(feat_bb, is_source=False), 'trg_bb')
            losses.update(trg_da_bb_loss)
        if self.da_neck_head != None:
            trg_da_neck_loss = add_prefix(self.da_neck_head(feat_neck, is_source=False), 'trg_neck')
            losses.update(trg_da_neck_loss)
        if self.da_pred_head != None:
            trg_da_pred_loss = add_prefix(self.da_pred_head(feat_pred, is_source=False), 'trg_pred')
            losses.update(trg_da_pred_loss)
        if self.da_output_head != None:
            trg_da_output_loss = add_prefix(self.da_output_head(pred_maps, is_source=False), 'trg_output')
            losses.update(trg_da_output_loss)
        # cate loss
        # feat_neck or feat_pred ?
        if self.enable_category_loss:
            trg_anchor, trg_weight = self.cal_anchor_and_weight(
                feat_maps=feat_pred, pred_maps=pred_maps, sign='trg')

        # src
        feat_bb, feat_neck = model.extract_feat_mine(source_img)
        feat_pred, pred_maps = model.bbox_head.return_feat_pred(feat_neck)
        if self.da_backbone_head != None:
            src_da_bb_loss = add_prefix(self.da_backbone_head(feat_bb, is_source=True), 'src_bb')
            losses.update(src_da_bb_loss)
        if self.da_neck_head != None:
            src_da_neck_loss = add_prefix(self.da_neck_head(feat_neck, is_source=True), 'src_neck')
            losses.update(src_da_neck_loss)
        if self.da_pred_head != None:
            src_da_pred_loss = add_prefix(self.da_pred_head(feat_pred, is_source=True), 'src_pred')
            losses.update(src_da_pred_loss)
        if self.da_output_head != None:
            src_da_output_loss = add_prefix(self.da_output_head(pred_maps, is_source=True), 'src_output')
            losses.update(src_da_output_loss)
        # cate loss
        # feat_neck or feat_pred ?
        if self.enable_category_loss:
            # src_anchor, src_weight = self.cal_anchor_and_weight(
            #     feat_maps=feat_pred, pred_maps=pred_maps, sign='src')
            src_anchor, src_weight = self.cal_anchor_and_weight(
                feat_maps=feat_pred, pred_maps=model.bbox_head.return_target_maps_list(
                pred_maps, gt_bboxes, gt_labels, source_img_metas), sign='src')
        # head loss
        loss = model.bbox_head.forward_train(
            feat_neck, source_img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(loss) 

        # category loss
        if self.enable_category_loss:
            for i in range(3):
                loss = self.category_loss(src_anchor[i], src_weight[i], trg_anchor[i], trg_weight[i], layer_lvl=i)
                losses.update(add_prefix(loss, f'anchor_{i}'))
        
        ### !!!
        # update memory bank
        if self.enable_category_loss:
            for i in range(len(src_anchor)):
                self.update_memory_bank(src_anchor[i], src_weight[i], i, 'src')
                self.update_memory_bank(trg_anchor[i], trg_weight[i], i, 'trg')
        return losses

    def gradient_penalty(self, src_feat, trg_feat):
        gamma = torch.rand(1).cuda()
        total_feat = []
        for i in range(len(src_feat)):
            total_feat.append(gamma * src_feat[i] + (1-gamma) * trg_feat[i])
        # total_feat = gamma * src_feat + (1-gamma) * trg_feat
        loss_total_feat = self.da_head(total_feat, is_source=False)
        gradient_gp = torch.autograd.grad(loss_total_feat, total_feat)
        gradient_norm = torch.norm(gradient_gp)
        loss_gp = torch.pow((gradient_norm - 1), 2)
        return loss_gp

    '''
    v3 将mean改为了max, 三个anchor框中选择置信度最高的对应的cls作为feat的cls, 用于后续操作
    v4 加入了conf_mask, conf<0.5的weight变为0, feat不会被考虑,  以及将weight调整为mean后^2,放大高置信度特征的权重
    v5 自定义了loss, 基于相似度的loss。
    v6 在v5的基础上, 除了conf mask之外, 额外考虑了cls的mask。
    v7 在srcmemorybank和trgmemorybank的基础上, 加入了src_weight和trg_weight
    v8 更新了update mb和weight, 防止weight=0的anchor也更新weight和mb
    v9 src的mask从gt来, 而不是pseudo label
    v10 解耦三个ahchor的输出
    v11 同时考虑conf和cate的masks
    '''
    def cal_anchor_and_weight(self, feat_maps, pred_maps, sign='src'):
        # feat.shape 
        #   [ torch.Size([8, 512, 19, 18]), 
        #     torch.Size([8, 256, 38, 36]), 
        #     torch.Size([8, 128, 76, 72])]
        # pred.shape  [(N,33,H,W)*3]
        anchor_list = []
        weight_list = []
        eps = torch.finfo(torch.float).eps
        for i in range(3):
            N, C, H, W = feat_maps[i].shape
            # print(feat_maps[i].shape)
            # print(pred_maps[i].shape)
            # input('ck')
            feat_map_per_level = feat_maps[i].permute(0, 2, 3, 1) # (N, H, W, C)
            if sign == 'trg':
                pred_map_per_level = pred_maps[i].permute(0, 2, 3, 1).reshape(N, H, W, 3, 5+self.num_classes)
                pred_map_per_level_conf = (pred_map_per_level[...,4]).sigmoid() # (N, H, W, 3)
                pred_map_per_level_conf, index = pred_map_per_level_conf.max(dim=-1) # (N, H, W)
                # hyper parameter test
                pred_map_per_level_conf = pred_map_per_level_conf.unsqueeze(-1) # (N, H, W, 1)
                conf_mask = (pred_map_per_level_conf > self.conf_thres) # (N, H, W, 1)
                pred_map_per_level_cls = pred_map_per_level[...,5:].sigmoid() # (N, H, W, 3, 6)
                index = index.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,self.num_classes]) # (N, H, W, 1, 6)
                # Get the probability prediction corresponding to the anchor most likely to find the object
                pred_map_per_level_cls = torch.gather(pred_map_per_level_cls, -2, index).squeeze_(-2) # (N, H, W, 6)
                # hyper parameter test
                pred_map_mask = (pred_map_per_level_cls > self.pred_thres) # (N, H, W, 6)
                # if conf < 0.5 and cls < 0.5, then cls is not dependable
                if self.cfa_v == 9: 
                    pred_map_per_level_cls = conf_mask * pred_map_mask * pred_map_per_level_cls # (N, H, W, 6)
                elif self.cfa_v == 11:
                    pred_map_per_level_cls = conf_mask * pred_map_per_level_conf * pred_map_mask * pred_map_per_level_cls  # (N, H, W, 6)
                elif self.cfa_v == 12:
                    pred_map_per_level_cls = conf_mask * pred_map_per_level_conf * pred_map_per_level_cls  # (N, H, W, 6)
                elif self.cfa_v == 13:
                    pred_map_per_level_cls = pred_map_mask * pred_map_per_level_cls   # (N, H, W, 6)
                elif self.cfa_v == 14:
                    pred_map_per_level_cls = conf_mask * pred_map_per_level_cls   # (N, H, W, 6)
                ### use a as the weight beteween 1 and each tensor, use the SGD to update the weight
                elif self.cfa_v == 15:
                    weight = self.loss_weight
                    conf_mask = conf_mask * torch.sigmoid(weight[0]) + (1 - torch.sigmoid(weight[0]))*torch.ones_like(conf_mask)
                    pred_map_mask = pred_map_mask * torch.sigmoid(weight[1]) + (1 - torch.sigmoid(weight[1]))*torch.ones_like(pred_map_mask)
                    pred_map_per_level_conf = pred_map_per_level_conf * torch.sigmoid(weight[2]) + (1 - torch.sigmoid(weight[2]))*torch.ones_like(pred_map_per_level_conf)              
                    pred_map_per_level_cls = conf_mask * pred_map_per_level_conf * pred_map_mask * pred_map_per_level_cls  # (N, H, W, 6)
                elif self.cfa_v == 16:
                    weight = [torch.clamp(self.loss_weight[i], min=0, max=1) for i in range(len(self.loss_weight))]
                    conf_mask = conf_mask * weight[0] + (1 - weight[0])*torch.ones_like(conf_mask)
                    pred_map_mask = pred_map_mask * weight[1] + (1 - weight[1])*torch.ones_like(pred_map_mask)
                    pred_map_per_level_conf = pred_map_per_level_conf * weight[2] + (1 - weight[2])*torch.ones_like(pred_map_per_level_conf)              
                    pred_map_per_level_cls = conf_mask * pred_map_per_level_conf * pred_map_mask * pred_map_per_level_cls  # (N, H, W, 6)
            else:
                pred_map_per_level = pred_maps[i].reshape(N, H, W, 3, -1)
                conf_mask = pred_map_per_level[...,4].sum(dim=-1).unsqueeze(-1).clamp(-1,1)
                pred_map_mask = pred_map_per_level[...,5:].sum(dim=-2).clamp(-1,1)
                pred_map_per_level_cls = conf_mask * pred_map_mask

            anchor = []
            weight = torch.zeros([self.num_classes,1], device=pred_map_per_level_cls.device, dtype=torch.float)
            for j in range(self.num_classes):
                pred_j = pred_map_per_level_cls[...,j] # (N, H, W)
                anchor.append((feat_map_per_level * pred_j.unsqueeze(-1)).sum(dim=[0,1,2]) / (
                    pred_j.sum() + eps ) )
                weight_j = pred_j.sum() / ((pred_j > 0).sum() + eps) 
                weight[j] = weight_j 
            anchor = torch.stack(anchor, dim=0)
            anchor_list.append(anchor) 
            weight_list.append(weight)
            ### !!!
            # self.update_memory_bank(anchor, weight, i, sign)
        return anchor_list, weight_list

    def cal_anchor_and_weight_v10(self, feat_maps, pred_maps, sign='src'):
        # feat.shape 
        #   [ torch.Size([8, 512, 19, 18]), 
        #     torch.Size([8, 256, 38, 36]), 
        #     torch.Size([8, 128, 76, 72])]
        # pred.shape  [(N,33,H,W)*3]
        anchor_list = []
        weight_list = []
        eps = torch.finfo(torch.float).eps
        conf_thres =  self.conf_thres
        pred_thres =  self.pred_thres
        for i in range(3):
            N, C, H, W = feat_maps[i].shape
            feat_map_per_level = feat_maps[i].permute(0, 2, 3, 1) # (N, H, W, C)
            if sign == 'trg':
                pred_map_per_level = pred_maps[i].permute(0, 2, 3, 1).reshape(N, H, W, 3, -1)
                pred_map_per_level_conf = (pred_map_per_level[...,4]).sigmoid() # (N, H, W, 3)
                # pred_map_per_level_conf, index = pred_map_per_level_conf.max(dim=-1) # (N, H, W)
                # hyper parameter test
                conf_mask = (pred_map_per_level_conf > conf_thres).unsqueeze(-1) # (N, H, W, 3, 1)
                # index = index.unsqueeze(-1).unsqueeze(-1).repeat([1,1,1,1,6]) # (N, H, W, 1, 6)
                pred_map_per_level_cls = pred_map_per_level[...,5:].sigmoid() # (N, H, W, 3, 6)
                # Get the probability prediction corresponding to the anchor most likely to find the object
                # pred_map_per_level_cls = torch.gather(pred_map_per_level_cls, -2, index).squeeze_(-2) # (N, H, W, 6)
                # hyper parameter test
                pred_map_mask = (pred_map_per_level_cls > pred_thres) # (N, H, W, 3, 6)
                # if conf < 0.5 and cls < 0.5, then cls is not dependable
                pred_map_per_level_cls = conf_mask * pred_map_mask * pred_map_per_level_cls # (N, H, W, 3, 6)
            else:
                pred_map_per_level = pred_maps[i].reshape(N, H, W, 3, -1) 
                conf_mask = pred_map_per_level[...,4].unsqueeze(-1) # (N, H, W, 3, 1)
                assert conf_mask.max() <=1
                pred_map_mask = pred_map_per_level[...,5:] # (N, H, W, 3, 6)
                assert pred_map_mask.max() <=1
                pred_map_per_level_cls = conf_mask * pred_map_mask # (N, H, W, 3, 6)

            anchor = []
            weight = torch.zeros([self.num_classes,1], device=pred_map_per_level_cls.device, dtype=torch.float)
            for j in range(self.num_classes):
                pred_j = pred_map_per_level_cls[...,j] # (N, H, W, 3)
                anchor.append((feat_map_per_level.unsqueeze(-2) * pred_j.unsqueeze(-1)).sum(dim=[0,1,2,3]) / (
                    pred_j.sum() + eps ) )
                weight_j = pred_j.sum() / ((pred_j > 0).sum() + eps) 
                weight[j] = weight_j 
            anchor = torch.stack(anchor, dim=0)
            anchor_list.append(anchor) 
            weight_list.append(weight)
        return anchor_list, weight_list
  

    ### !!!
    @torch.no_grad()
    def update_memory_bank(self, anchor, weight, layer_lvl, sign='src'):
        ### shape
        # anchor.shape: [6, C] 
        # weight.shape: [6, 1]
        alpha = self.alpha_mb 
        if sign == 'src':
            for i in range(self.num_classes):
                if weight[i] > 0:
                    self.src_memory_bank[layer_lvl][i] = self.src_memory_bank[layer_lvl][i]*(1-alpha*weight[i]) + alpha*weight[i]*anchor[i] # [6, C]
                    self.src_weight[layer_lvl][i] = self.src_weight[layer_lvl][i]*(1-alpha*weight[i]) + alpha*weight[i]*weight[i]
        if sign == 'trg':
            for i in range(self.num_classes):
                if weight[i] > 0:
                    self.trg_memory_bank[layer_lvl][i] = self.trg_memory_bank[layer_lvl][i]*(1-alpha*weight[i]) + alpha*weight[i]*anchor[i] # [6, C]
                    self.trg_weight[layer_lvl][i] = self.trg_weight[layer_lvl][i]*(1-alpha*weight[i]) + alpha*weight[i]*weight[i]

        # if sign == 'src':
        #     for i in range(6):
        #         if weight[i] > 0:
        #             self.src_memory_bank[layer_lvl][i] = self.src_memory_bank[layer_lvl][i]*(1-alpha) + alpha*anchor[i] # [6, C]
        #             self.src_weight[layer_lvl][i] = self.src_weight[layer_lvl][i]*(1-alpha) + alpha*weight[i]
        # if sign == 'trg':
        #     for i in range(6):
        #         if weight[i] > 0:
        #             self.trg_memory_bank[layer_lvl][i] = self.trg_memory_bank[layer_lvl][i]*(1-alpha) + alpha*anchor[i] # [6, C]
        #             self.trg_weight[layer_lvl][i] = self.trg_weight[layer_lvl][i]*(1-alpha) + alpha*weight[i]
    
    def category_loss(self, src_anchor, src_weight, trg_anchor, trg_weight, layer_lvl=0):
        K, C = src_anchor.shape
        # eps = torch.finfo(torch.float).eps
        # loss_intra = torch.zeros([1], dtype=torch.float, device=src_anchor.device)
        # weight = torch.zeros([1], dtype=torch.float, device=src_anchor.device)

        ### sim matrix
        # sim_matrix_ss = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        # sim_matrix_st = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        # sim_matrix_ts = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        # sim_matrix_tt = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        if self.local_iter % 154 == 1:
            sim_matrix_st_memory = torch.zeros([K,K], dtype=torch.float, device=src_anchor.device)
            for i in range(K):
                for j in range(K):
                    # sim_matrix_st[i,j] = self.similarity(src_anchor[i], self.trg_memory_bank[layer_lvl][j])
                    # sim_matrix_ss[i,j] = self.similarity(src_anchor[i], self.src_memory_bank[layer_lvl][j])
                    # sim_matrix_tt[i,j] = self.similarity(trg_anchor[i], self.trg_memory_bank[layer_lvl][j])
                    # sim_matrix_ts[i,j] = self.similarity(trg_anchor[i], self.src_memory_bank[layer_lvl][j])
                    sim_matrix_st_memory[i,j] = self.similarity(self.trg_memory_bank[layer_lvl][i], self.src_memory_bank[layer_lvl][j])
            torch.set_printoptions(
                precision=4,
                # threshold=1000,
                # edgeitems=3,
                linewidth=150,
                profile=None,
                sci_mode=False
            )
            mmcv.print_log(f'layer_lvl: {layer_lvl} ', 'mmdet')
            mmcv.print_log(f'src_weight_{layer_lvl}: {self.src_weight[layer_lvl].t()} ', 'mmdet')
            mmcv.print_log(f'trg_weight_{layer_lvl}: {self.trg_weight[layer_lvl].t()} ', 'mmdet')
            mmcv.print_log(f'sim_matrix_st_memory:\n {sim_matrix_st_memory} ', 'mmdet')
        '''
        ### norm
        # sim_matrix_ss_norm = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        # sim_matrix_st_norm = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        # sim_matrix_tt_norm = torch.zeros([6,6], dtype=torch.float, device=src_anchor.device)
        # for i in range(K):
        #     for j in range(K):
        #         sim_matrix_st_norm[i,j] = torch.norm(src_anchor[i]-self.trg_memory_bank[layer_lvl][j])
        #         sim_matrix_ss_norm[i,j] = torch.norm(src_anchor[i]-self.src_memory_bank[layer_lvl][j])
        #         sim_matrix_tt_norm[i,j] = torch.norm(trg_anchor[i]-self.trg_memory_bank[layer_lvl][j])
        # print('sim_matrix_st')
        # print(sim_matrix_st_norm)
        # print('sim_matrix_ss_norm')
        # print(sim_matrix_ss_norm)
        # print('sim_matrix_tt_norm')
        # print(sim_matrix_tt_norm)
        # ### intra
        # for i in range(K):
        #     for j in range(K):
        #         loss_intra += src_weight[i] *  torch.norm(src_anchor[i] - trg_anchor[j]) * trg_weight[j]
        #         weight += src_weight[i] * trg_weight[j]
        # loss_intra = loss_intra / (weight + eps)
        '''
        ### inter
        loss_inter_st = self.inter_loss(src_anchor, src_weight, self.trg_memory_bank[layer_lvl], self.trg_weight[layer_lvl]) 
        # loss_inter_ts = self.inter_loss(trg_anchor, trg_weight, self.src_memory_bank[layer_lvl], self.src_weight[layer_lvl])
        loss_inter_ss = self.inter_loss(src_anchor, src_weight, self.src_memory_bank[layer_lvl], self.src_weight[layer_lvl]) 
        loss_inter_tt = self.inter_loss(trg_anchor, trg_weight, self.trg_memory_bank[layer_lvl], self.trg_weight[layer_lvl])
        losses = dict()
        losses['loss_inter_st'] = loss_inter_st * self.category_weight
        # losses['loss_inter_ts'] = loss_inter_ts * self.category_weight
        losses['loss_inter_ss'] = loss_inter_ss * self.category_weight
        losses['loss_inter_tt'] = loss_inter_tt * self.category_weight
        return losses

    def inter_loss(self, src_anchor, src_weight, trg_anchor, trg_weight):
        # input(src_anchor.shape)
        eps = torch.finfo(torch.float).eps
        src_anchor = src_anchor / (torch.norm(src_anchor, dim=-1, keepdim=True) + eps)
        trg_anchor = trg_anchor / (torch.norm(trg_anchor, dim=-1, keepdim=True) + eps)
        src_anchor_weight = src_anchor * src_weight
        trg_anchor_weight = trg_anchor * trg_weight
        # input(src_anchor.shape)
        # input(src_weight.shape)
        # input(trg_anchor.shape)
        # input(trg_weight.shape)
        matrix = src_anchor_weight @ trg_anchor_weight.t()
        loss = matrix.sum() - 2 * matrix.trace() 
        return loss
    
    def inter_loss_old(self, src_anchor, src_weight, trg_anchor, trg_weight):
        K, C = src_anchor.shape
        eps = torch.finfo(torch.float).eps
        loss = torch.zeros([1], dtype=torch.float, device=src_anchor.device)
        for i in range(K):
            for j in range(K):
                if i != j :
                    loss += self.similarity(src_anchor[i], trg_anchor[j]) * src_weight[i] * trg_weight[j] 
                else:
                    loss -= self.similarity(src_anchor[i], trg_anchor[j]) * src_weight[i] * trg_weight[j]
        return loss

    def similarity(self, anchorone, anchortwo):
        eps = torch.finfo(torch.float).eps
        return  (anchorone @ anchortwo) / (torch.norm(anchorone) * torch.norm(anchortwo) + eps)
        
    # def ease_loss(self, layer_lvl=0, lossweight=1):
    def ease_loss(self, src_anchor, src_weight, trg_anchor, trg_weight, layer_lvl=0):
        trg_memory_bank = self.trg_memory_bank[layer_lvl] 
        trg_mb_mask = (trg_memory_bank.sum(dim=1)>0)
        src_memory_bank = self.src_memory_bank[layer_lvl] 
        src_mb_mask = (src_memory_bank.sum(dim=1)>0)
        mask_total = (trg_mb_mask + trg_weight.squeeze() + src_mb_mask + trg_weight.squeeze())
        classnum = (mask_total>0).sum()
        loss = torch.tensor(0.).cuda()
        if classnum>=2:
            with torch.no_grad():
                feat_cat = []
                for i in range(6):
                    if src_weight[i]>0: feat_cat.append(src_anchor[i])
                    if trg_weight[i]>0: feat_cat.append(trg_anchor[i])
                    if src_mb_mask[i]>0: feat_cat.append(src_memory_bank[i])
                    if trg_mb_mask[i]>0: feat_cat.append(trg_memory_bank[i])
                feat_cat = torch.stack(feat_cat, dim=0).clone().detach() 
            N, C = feat_cat.shape
            feat_cat = feat_cat / (feat_cat.norm(dim=-1, keepdim=True) + self.eps)
            sim_matrix = 2*(1-(feat_cat @ feat_cat.t()))
            sim_matrix = torch.exp(-sim_matrix)
            D = torch.diag(1/(sim_matrix.sum(dim=-1).sqrt()+self.eps))
            sim_matrix = D @ sim_matrix @ D
            dis_matrix = torch.ones([N,N]).cuda()/N - torch.eye(N).cuda()
            feat_cat.unsqueeze_(-1).unsqueeze_(-1)
            pred_map_cat = self.model.bbox_head.convs_pred[layer_lvl](feat_cat)
            loss = -(pred_map_cat.squeeze().t() @ (dis_matrix - sim_matrix) @ pred_map_cat.squeeze()).trace() * self.ease_weight
        return ({f'loss_ease_{layer_lvl}':loss})

    # def init_linear(self, cin, cout):
    #     self.beta = 1
    #     self.q = 0.5
    #     self.s = 2
    #     self.lr = 0.2
    #     # Initialize linear as an orthonormal matrix
    #     cout, cin = self.linear_layer.weight.shape
    #     index = torch.arange(cout) 
    #     self.linear_layer.weight[index, index] = 1
    #     self.M_his = torch.zeros_like(self.linear_layer)
    
    # def update_linear(self, grad):
    #     unity = self.linear_layer.weight
    #     M = self.beta * self.M_his - grad.t()   
    #     MX = torch.mm(M, unity)
    #     XMX = torch.mm(unity, MX)
    #     XXMX = torch.mm(unity.t(), XMX)
    #     W_hat = MX - 0.5 * XXMX
    #     W = W_hat - W_hat.t()
    #     t = self.q * 2 / (W.norm() + self.eps)                    
    #     alpha = min(t, self.lr)
    #     Y = self.Cayley_loop(unity.t(), W, M, alpha)
    #     self.M_his = torch.mm(W, unity.t()) # n-by-p
    #     self.linear_layer.weight = Y

    # def Cayley_loop(self, X, W, tan_vec, t): # 
    #     [n, p] = X.size()
    #     Y = X + t * tan_vec
    #     for i in range(2):
    #         Y = X + t * torch.matmul(W, 0.5*(X+Y))
    #     return Y.t()

    # aux head pred loss
    def cal_aux_cls_loc_loss(self, feat, is_source=True):
        # feat.shape 
        '''
        # [   torch.Size([8, 512, 15, 19]), 
        #     torch.Size([8, 256, 30, 38]), 
        #     torch.Size([8, 128, 60, 76])]
        '''
        assert self.auxiliary_head_num != 0
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
            # print(losses)
            # input(f'=.=, {is_source}')
        return losses 
    
    def simple_test(self, img, img_metas, rescale=False):
        return self.get_model().simple_test(img, img_metas, rescale)
    
    def aug_test(self, imgs, img_metas, rescale=False):
        return self.get_model().aug_test(imgs, img_metas, rescale)


    # train_step for lossW
    '''
    def train_step(self, data, optimizer):
        """The iteration step during training.

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
        """
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


if __name__ == '__main__':
    dev = "cuda"
    torch.cuda.set_device(7)
    from mmcv.utils import Config
    from mmdet.models import build_detector, build_detector_mine
    import os
    cfg = Config.fromfile('/home/yebh/mmdetection/configs/uda/uda_yolov3_T2L_SCL-75k75k0-003-T07.py')
    cfg.work_dir='work_dirs/test'
    model = build_detector_mine(
        cfg,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    # for i in range(3):
    #     model.trg_memory_bank[i] += torch.rand([6,1]).cuda() 
    #     model.src_memory_bank[i] += torch.rand([6,1]).cuda() 
    # print(model.trg_memory_bank)
    # print(model.src_memory_bank)
    # model.ease_loss(layer_lvl=1)
    src_image = torch.randint(255, [1,3,640,640]).float().cuda()
    trg_image = torch.randint(255, [1,3,640,640]).float().cuda()
    src_feat = model.extract_feat(src_image)[-3:]
    # input(len(src_feat))
    # input([ele.shape for ele in src_feat])
    trg_feat = model.extract_feat(trg_image)[-3:]
    print(model.gradient_penalty(src_feat, trg_feat))
    
