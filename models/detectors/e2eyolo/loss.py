import torch
import torch.nn.functional as F
from .matcher import TaskAlignedAssigner
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        # loss weight
        self.loss_obj_weight = cfg['loss_obj_weight']
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


    def loss_objectness(self, pred_obj, gt_obj):
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

        return loss_obj
    

    def loss_classes(self, pred_cls, gt_label):
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_label, reduction='none')

        return loss_cls


    def loss_bboxes(self, pred_box, gt_box):
        # regression loss
        ious = get_ious(pred_box,
                        gt_box,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_box = 1.0 - ious

        return loss_box


    def __call__(self, outputs, targets):        
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
        anchors = torch.cat(outputs['anchors'], dim=0)
        num_anchors = anchors.shape[0]

        # preds: [B, M, C]
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)
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
                gt_label = cls_preds.new_zeros((1, num_anchors,))                  #[1, M,]
                gt_score = cls_preds.new_zeros((1, num_anchors, self.num_classes)) #[1, M, C]
                gt_box = cls_preds.new_zeros((1, num_anchors, 4))                  #[1, M, 4]
            else:
                tgt_labels = tgt_labels[None, :, None]      # [1, Mp, 1]
                tgt_boxs = tgt_boxs[None]                   # [1, Mp, 4]
                (
                    gt_label,   #[1, M]
                    gt_box,     #[1, M, 4]
                    gt_score,   #[1, M, C]
                    fg_mask,    #[1, M,]
                    _
                ) = self.matcher(
                    pd_scores = torch.sqrt(obj_preds[batch_idx:batch_idx+1].sigmoid() * \
                                           cls_preds[batch_idx:batch_idx+1].sigmoid()).detach(), 
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
        gt_label_targets = torch.cat(gt_label_targets, 0).view(-1)                    # [BM,]
        gt_score_targets = torch.cat(gt_score_targets, 0).view(-1, self.num_classes)  # [BM, C]
        gt_bbox_targets = torch.cat(gt_bbox_targets, 0).view(-1, 4)                   # [BM, 4]

        obj_targets = fg_masks.unsqueeze(-1)        # [M, 1]
        cls_targets = gt_score_targets[fg_masks]    # [Mp, C]
        box_targets = gt_bbox_targets[fg_masks]     # [Mp, 4]
        num_fgs = fg_masks.sum()
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # obj loss
        loss_obj = self.loss_objectness(obj_preds.view(-1, 1), obj_targets.float())
        loss_obj = loss_obj.sum() / num_fgs
        
        # cls loss
        cls_preds_pos = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.loss_classes(cls_preds_pos, cls_targets)
        loss_cls = loss_cls.sum() / num_fgs

        # regression loss
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets)
        loss_box = loss_box.sum() / num_fgs

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        loss_dict = dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict
    

def build_criterion(cfg, device, num_classes):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        num_classes=num_classes
        )

    return criterion


if __name__ == "__main__":
    pass