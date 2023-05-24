import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import TaskAlignedAssigner, Yolov5Matcher
from utils.box_ops import bbox_iou, get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 num_classes=80,
                 warmup_epoch=1):
        # ------------------ Basic Parameters ------------------
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.warmup_epoch = warmup_epoch
        # ------------------ Loss Parameters ------------------
        ## loss function
        self.cls_lossf = ClassificationLoss(cfg, reduction='none')
        self.reg_lossf = RegressionLoss(num_classes)
        ## loss coeff
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_iou_weight = cfg['loss_iou_weight']
        # ------------------ Label Assigner ------------------
        matcher_config = cfg['matcher']
        ## matcher-1
        self.fixed_matcher = Yolov5Matcher(
            num_classes=num_classes, 
            num_anchors=3, 
            anchor_size=cfg['anchor_size'],
            anchor_theshold=matcher_config['anchor_thresh']
            )
        ## matcher-2
        self.dynamic_matcher = TaskAlignedAssigner(
            topk=matcher_config['topk'],
            num_classes=num_classes,
            alpha=matcher_config['alpha'],
            beta=matcher_config['beta']
            )


    def fixed_assignment_loss(self, outputs, targets):
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        fmp_sizes = outputs['fmp_sizes']
        (
            gt_objectness, 
            gt_classes, 
            gt_bboxes,
            ) = self.fixed_matcher(fmp_sizes=fmp_sizes, 
                                   fpn_strides=fpn_strides, 
                                   targets=targets)
        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)    # [BM, C]
        pred_box = torch.cat(outputs['pred_box'], dim=1).view(-1, 4)                   # [BM, 4]
       
        gt_objectness = gt_objectness.view(-1).to(device).float()               # [BM,]
        gt_classes = gt_classes.view(-1, self.num_classes).to(device).float()   # [BM, C]
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()                    # [BM, 4]

        pos_masks = (gt_objectness > 0)
        num_fgs = pos_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # box loss
        ious = get_ious(pred_box[pos_masks],
                        gt_bboxes[pos_masks],
                        box_mode="xyxy",
                        iou_type='giou')
        loss_box = 1.0 - ious
        loss_box = loss_box.sum() / num_fgs
        
        # cls loss
        gt_classes[pos_masks] = gt_classes[pos_masks] * ious.unsqueeze(-1).clamp(0.)
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_classes, reduction='none')
        loss_cls = loss_cls.sum() / num_fgs

        # total loss
        losses = self.loss_cls_weight * loss_cls + \
                 self.loss_iou_weight * loss_box

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict


    def dynamic_assignment_loss(self, outputs, targets):
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        anchors = outputs['anchors']
        anchors = torch.cat(anchors, dim=0)
        num_anchors = anchors.shape[0]

        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        
        # label assignment
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
                    _,
                    gt_box,     #[1, M, 4]
                    gt_score,   #[1, M, C]
                    fg_mask,    #[1, M,]
                    _
                ) = self.dynamic_matcher(
                    pd_scores = cls_preds[batch_idx:batch_idx+1].detach().sigmoid(), 
                    pd_bboxes = box_preds[batch_idx:batch_idx+1].detach(),
                    anc_points = anchors[..., :2],
                    gt_labels = tgt_labels,
                    gt_bboxes = tgt_boxs
                    )
            gt_score_targets.append(gt_score)
            gt_bbox_targets.append(gt_box)
            fg_masks.append(fg_mask)

        # List[B, 1, M, C] -> Tensor[B, M, C] -> Tensor[BM, C]
        fg_masks = torch.cat(fg_masks, 0).view(-1)                                    # [BM,]
        gt_score_targets = torch.cat(gt_score_targets, 0).view(-1, self.num_classes)  # [BM, C]
        gt_bbox_targets = torch.cat(gt_bbox_targets, 0).view(-1, 4)                   # [BM, 4]
        
        # cls loss
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.cls_lossf(cls_preds, gt_score_targets)

        # reg loss
        bbox_weight = gt_score_targets[fg_masks].sum(-1, keepdim=True)                 # [BM, 1]
        box_preds = box_preds.view(-1, 4)                                              # [BM, 4]
        loss_iou = self.reg_lossf(
            pred_boxs = box_preds,
            gt_boxs = gt_bbox_targets,
            bbox_weight = bbox_weight,
            fg_masks = fg_masks
            )

        num_fgs = gt_score_targets.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # normalize loss
        loss_cls = loss_cls.sum() / num_fgs
        loss_iou = loss_iou.sum() / num_fgs

        # total loss
        losses = loss_cls * self.loss_cls_weight + \
                 loss_iou * self.loss_iou_weight
        loss_dict = dict(
                loss_cls = loss_cls,
                loss_iou = loss_iou,
                losses = losses
        )

        return loss_dict


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
        # Fixed LA stage
        if epoch < self.warmup_epoch:
            return self.fixed_assignment_loss(outputs, targets)
        # Switch to Dynamic LA stage
        elif epoch == self.warmup_epoch:
            print('Switch to Dynamic Label Assignment.')
            return self.dynamic_assignment_loss(outputs, targets)
        # Dynamic LA stage
        else:
            return self.dynamic_assignment_loss(outputs, targets)
    

class ClassificationLoss(nn.Module):
    def __init__(self, cfg, reduction='none'):
        super(ClassificationLoss, self).__init__()
        self.cfg = cfg
        self.reduction = reduction


    def binary_cross_entropy(self, pred_logits, gt_score):
        loss = F.binary_cross_entropy_with_logits(
            pred_logits.float(), gt_score.float(), reduction='none')

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


    def forward(self, pred_logits, gt_score):
        if self.cfg['cls_loss'] == 'bce':
            return self.binary_cross_entropy(pred_logits, gt_score)


class RegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(RegressionLoss, self).__init__()
        self.num_classes = num_classes


    def forward(self, pred_boxs, gt_boxs, bbox_weight, fg_masks):
        """
        Input:
            pred_boxs: (Tensor) [BM, 4]
            anchors: (Tensor) [BM, 2]
            gt_boxs: (Tensor) [BM, 4]
            bbox_weight: (Tensor) [BM, 1]
            fg_masks: (Tensor) [BM,]
            strides: (Tensor) [BM, 1]
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


def build_criterion(cfg, device, num_classes, warmup_epoch=1):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        num_classes=num_classes,
        warmup_epoch=warmup_epoch,
        )

    return criterion


if __name__ == "__main__":
    pass