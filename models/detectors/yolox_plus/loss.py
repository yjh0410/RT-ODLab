import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import TaskAlignedAssigner
from utils.box_ops import bbox_iou



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        # loss
        self.cls_lossf = ClassificationLoss(cfg)
        self.reg_lossf = RegressionLoss(num_classes)
        # loss weight
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_box_weight = cfg['loss_box_weight']
        # matcher
        matcher_config = cfg['matcher']
        self.matcher = TaskAlignedAssigner(
            topk=matcher_config['topk'],
            num_classes=num_classes,
            alpha=matcher_config['alpha'],
            beta=matcher_config['beta']
            )


    def __call__(self, outputs, targets, epoch=0):        
        """
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_regs']: List(Tensor) [B, M, 4*(reg_max+1)]
            outputs['pred_boxs']: List(Tensor) [B, M, 4]
            outputs['anchors']: List(Tensor) [M, 2]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            outputs['stride_tensor']: List(Tensor) [M, 1]
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        anchors = outputs['anchors']
        anchors = torch.cat(anchors, dim=0)
        num_anchors = anchors.shape[0]

        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        
        # label assignment
        gt_label_targets = []
        gt_score_targets = []
        gt_bbox_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)     # [Mp,]
            tgt_boxs = targets[batch_idx]["boxes"].to(device)        # [Mp, 4]

            # check target
            if len(tgt_labels) == 0 or tgt_boxs.max().item() == 0.:
                # There is no valid gt
                fg_mask = cls_preds.new_zeros(1, num_anchors).bool()               #[1, M,]
                gt_score = cls_preds.new_zeros((1, num_anchors, self.num_classes)) #[1, M, C]
                gt_box = cls_preds.new_zeros((1, num_anchors, 4))                  #[1, M, 4]
            else:
                tgt_labels = tgt_labels[None, :, None]      # [1, Mp, 1]
                tgt_boxs = tgt_boxs[None]                   # [1, Mp, 4]
                (
                    gt_label,
                    gt_box,     #[1, M, 4]
                    gt_score,   #[1, M, C]
                    fg_mask,    #[1, M,]
                    _
                ) = self.matcher(
                    pd_scores = cls_preds[batch_idx:batch_idx+1].detach().sigmoid(), 
                    pd_bboxes = box_preds[batch_idx:batch_idx+1].detach(),
                    anc_points = anchors,
                    gt_labels = tgt_labels,
                    gt_bboxes = tgt_boxs
                    )
            gt_label_targets.append(gt_label)
            gt_score_targets.append(gt_score)
            gt_bbox_targets.append(gt_box)
            fg_masks.append(fg_mask)

        # List[B, 1, M, C] -> Tensor[B, M, C] -> Tensor[BM, C]
        fg_masks = torch.cat(fg_masks, 0).view(-1)                                    # [BM,]
        gt_label_targets = torch.cat(gt_label_targets, 0).view(-1,)                   # [BM, 1]
        gt_score_targets = torch.cat(gt_score_targets, 0).view(-1, self.num_classes)  # [BM, C]
        gt_bbox_targets = torch.cat(gt_bbox_targets, 0).view(-1, 4)                   # [BM, 4]
       
        # cls loss
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.cls_lossf(cls_preds, gt_label_targets, gt_score_targets)

        # reg loss
        bbox_weight = gt_score_targets[fg_masks].sum(-1, keepdim=True)                 # [BM, 1]
        box_preds = box_preds.view(-1, 4)                                              # [BM, 4]
        loss_box = self.reg_lossf(box_preds, gt_bbox_targets, bbox_weight, fg_masks)
        
        # normalize loss
        gt_score_targets_sum = max(gt_score_targets.sum(), 1)
        loss_cls = loss_cls.sum() / gt_score_targets_sum
        loss_box = loss_box.sum() / gt_score_targets_sum

        # total loss
        losses = loss_cls * self.loss_cls_weight + \
                 loss_box * self.loss_box_weight
        
        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict
    

class ClassificationLoss(nn.Module):
    def __init__(self, cfg):
        super(ClassificationLoss, self).__init__()
        self.cfg = cfg


    def quality_focal_loss(self, pred_cls, gt_label, gt_score, beta=2.0):
        # Quality FocalLoss
        """
            pred_cls: (torch.Tensor): [N, C]
            gt_label: (torch.Tensor): [N,]
            gt_score: (torch.Tensor): [N, C]
        """
        gt_label = gt_label.long()
        gt_score = gt_score[:]
        gt_score = gt_score[torch.arange(gt_label.shape[0]), gt_label]

        pred_sigmoid = pred_cls.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_cls.shape)

        ce_loss = F.binary_cross_entropy_with_logits(
            pred_cls, zerolabel, reduction='none') * scale_factor.pow(beta)
        
        bg_class_ind = pred_cls.shape[-1]
        pos = ((gt_label >= 0) & (gt_label < bg_class_ind)).nonzero().squeeze(1)
        pos_label = gt_label[pos].long()

        scale_factor = gt_score[pos] - pred_sigmoid[pos, pos_label]

        ce_loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred_cls[pos, pos_label], gt_score[pos],
            reduction='none') * scale_factor.abs().pow(beta)

        return ce_loss
    

    def binary_cross_entropy(self, pred_logits, gt_score):
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, gt_score, reduction='none')

        return loss


    def forward(self, pred_logits, gt_label, gt_score):
        if self.cfg['cls_loss'] == 'bce':
            loss = self.binary_cross_entropy(pred_logits, gt_score)
        elif self.cfg['cls_loss'] == 'qfl':
            loss = self.quality_focal_loss(pred_logits, gt_label, gt_score)
            
        return loss


class RegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(RegressionLoss, self).__init__()
        self.num_classes = num_classes


    def forward(self, pred_boxs, gt_boxs, bbox_weight, fg_masks):
        """
        Input:
            pred_boxs: (Tensor) [BM, 4]
            gt_boxs: (Tensor) [BM, 4]
            bbox_weight: (Tensor) [BM, 1]
            fg_masks: (Tensor) [BM,]
        """
        # select positive samples mask
        num_pos = fg_masks.sum()

        if num_pos > 0:
            pred_boxs_pos = pred_boxs[fg_masks]
            gt_boxs_pos = gt_boxs[fg_masks]

            # iou loss
            ious = bbox_iou(pred_boxs_pos,
                            gt_boxs_pos,
                            xywh=False,
                            CIoU=True)
            loss_iou = (1.0 - ious) * bbox_weight
               
        else:
            loss_iou = pred_boxs.sum() * 0.

        return loss_iou


def build_criterion(cfg, device, num_classes):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        num_classes=num_classes
        )

    return criterion


if __name__ == "__main__":
    pass