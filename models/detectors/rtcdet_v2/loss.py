import torch
import torch.nn.functional as F

from utils.box_ops import bbox2dist, get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import TaskAlignedAssigner, AlignedSimOTA


class Criterion(object):
    def __init__(self, args, cfg, device, num_classes=80):
        self.cfg = cfg
        self.args = args
        self.device = device
        self.num_classes = num_classes
        self.max_epoch = args.max_epoch
        self.no_aug_epoch = args.no_aug_epoch
        self.use_ema_update = cfg['ema_update']
        # ---------------- Loss weight ----------------
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_box_weight = cfg['loss_box_weight']
        self.loss_dfl_weight = cfg['loss_dfl_weight']
        self.loss_box_aux    = cfg['loss_box_aux']
        # ---------------- Matcher ----------------
        matcher_config = cfg['matcher']
        ## TAL assigner
        self.tal_matcher = TaskAlignedAssigner(
            topk=matcher_config['tal']['topk'],
            alpha=matcher_config['tal']['alpha'],
            beta=matcher_config['tal']['beta'],
            num_classes=num_classes
            )
        ## SimOTA assigner
        self.ota_matcher = AlignedSimOTA(
            center_sampling_radius=matcher_config['ota']['center_sampling_radius'],
            topk_candidate=matcher_config['ota']['topk_candidate'],
            num_classes=num_classes
        )

    def __call__(self, outputs, targets, epoch=0):
        if epoch < self.args.max_epoch // 2:
            return self.ota_loss(outputs, targets)
        else:
            return self.tal_loss(outputs, targets)

    def ema_update(self, name: str, value, initial_value, momentum=0.9):
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new

    # ----------------- Loss functions -----------------
    def loss_classes(self, pred_cls, gt_score, gt_label=None, vfl=False):
        if vfl:
            assert gt_label is not None
            # compute varifocal loss
            alpha, gamma = 0.75, 2.0
            focal_weight = alpha * pred_cls.sigmoid().pow(gamma) * (1 - gt_label) + gt_score * gt_label
            bce_loss = F.binary_cross_entropy_with_logits(pred_cls, gt_score, reduction='none')
            loss_cls = bce_loss * focal_weight
        else:
            # compute bce loss
            loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_score, reduction='none')

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box, bbox_weight=None):
        # regression loss
        ious = get_ious(pred_box, gt_box, 'xyxy', 'giou')
        loss_box = 1.0 - ious

        if bbox_weight is not None:
            loss_box *= bbox_weight

        return loss_box

    def loss_dfl(self, pred_reg, gt_box, anchor, stride, bbox_weight=None):
        # rescale coords by stride
        gt_box_s = gt_box / stride
        anchor_s = anchor / stride

        # compute deltas
        gt_ltrb_s = bbox2dist(anchor_s, gt_box_s, self.cfg['reg_max'] - 1)

        gt_left = gt_ltrb_s.to(torch.long)
        gt_right = gt_left + 1

        weight_left = gt_right.to(torch.float) - gt_ltrb_s
        weight_right = 1 - weight_left

        # loss left
        loss_left = F.cross_entropy(
            pred_reg.view(-1, self.cfg['reg_max']),
            gt_left.view(-1),
            reduction='none').view(gt_left.shape) * weight_left
        # loss right
        loss_right = F.cross_entropy(
            pred_reg.view(-1, self.cfg['reg_max']),
            gt_right.view(-1),
            reduction='none').view(gt_left.shape) * weight_right

        loss_dfl = (loss_left + loss_right).mean(-1)
        
        if bbox_weight is not None:
            loss_dfl *= bbox_weight

        return loss_dfl

    def loss_bboxes_aux(self, pred_delta, gt_box, anchors, stride_tensors):
        gt_delta_tl = (anchors - gt_box[..., :2]) / stride_tensors
        gt_delta_rb = (gt_box[..., 2:] - anchors) / stride_tensors
        gt_delta = torch.cat([gt_delta_tl, gt_delta_rb], dim=1)
        loss_box_aux = F.l1_loss(pred_delta, gt_delta, reduction='none')

        return loss_box_aux

    # ----------------- Loss with TAL assigner -----------------
    def tal_loss(self, outputs, targets, epoch=0):
        """ Compute loss with TAL assigner """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        anchors = torch.cat(outputs['anchors'], dim=0)
        num_anchors = anchors.shape[0]
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        reg_preds = torch.cat(outputs['pred_reg'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # --------------- label assignment ---------------
        gt_label_targets = []
        gt_score_targets = []
        gt_bbox_targets = []
        fg_masks = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                # There is no valid gt
                fg_mask = cls_preds.new_zeros(1, num_anchors).bool()               #[1, M,]
                gt_label = cls_preds.new_zeros((1, num_anchors,))                  #[1, M,]
                gt_score = cls_preds.new_zeros((1, num_anchors, self.num_classes)) #[1, M, C]
                gt_box = cls_preds.new_zeros((1, num_anchors, 4))                  #[1, M, 4]
            else:
                tgt_labels = tgt_labels[None, :, None]      # [1, Mp, 1]
                tgt_bboxes = tgt_bboxes[None]                   # [1, Mp, 4]
                (
                    gt_label,   #[1, M]
                    gt_box,     #[1, M, 4]
                    gt_score,   #[1, M, C]
                    fg_mask,    #[1, M,]
                    _
                ) = self.tal_matcher(
                    pd_scores = cls_preds[batch_idx:batch_idx+1].detach().sigmoid(), 
                    pd_bboxes = box_preds[batch_idx:batch_idx+1].detach(),
                    anc_points = anchors,
                    gt_labels = tgt_labels,
                    gt_bboxes = tgt_bboxes
                    )
            gt_label_targets.append(gt_label)
            gt_score_targets.append(gt_score)
            gt_bbox_targets.append(gt_box)
            fg_masks.append(fg_mask)

        # List[B, 1, M, C] -> Tensor[B, M, C] -> Tensor[BM, C]
        fg_masks = torch.cat(fg_masks, 0).view(-1)                                    # [BM,]
        gt_score_targets = torch.cat(gt_score_targets, 0).view(-1, self.num_classes)  # [BM, C]
        gt_bbox_targets = torch.cat(gt_bbox_targets, 0).view(-1, 4)                   # [BM, 4]
        gt_label_targets = torch.cat(gt_label_targets, 0).view(-1)                    # [BM,]
        gt_label_targets = torch.where(fg_masks > 0, gt_label_targets, torch.full_like(gt_label_targets, self.num_classes))
        gt_labels_one_hot = F.one_hot(gt_label_targets.long(), self.num_classes + 1)[..., :-1]
        bbox_weight = gt_score_targets[fg_masks].sum(-1)
        num_fgs = max(gt_score_targets.sum(), 1)

        # average loss normalizer across all the GPUs
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = max(num_fgs / get_world_size(), 1.0)

        # update loss normalizer with EMA
        if self.use_ema_update:
            normalizer = self.ema_update("loss_normalizer", max(num_fgs, 1), 100)
        else:
            normalizer = num_fgs

        # ------------------ Classification loss ------------------
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.loss_classes(cls_preds, gt_score_targets, gt_labels_one_hot, vfl=False)
        loss_cls = loss_cls.sum() / normalizer

        # ------------------ Regression loss ------------------
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        box_targets_pos = gt_bbox_targets[fg_masks]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets_pos, bbox_weight)
        loss_box = loss_box.sum() / normalizer

        # ------------------ Distribution focal loss  ------------------
        ## process anchors
        anchors = anchors[None].repeat(bs, 1, 1).view(-1, 2)
        ## process stride tensors
        strides = torch.cat(outputs['stride_tensor'], dim=0)
        strides = strides.unsqueeze(0).repeat(bs, 1, 1).view(-1, 1)
        ## fg preds
        reg_preds_pos = reg_preds.view(-1, 4*self.cfg['reg_max'])[fg_masks]
        anchors_pos = anchors[fg_masks]
        strides_pos = strides[fg_masks]
        ## compute dfl
        loss_dfl = self.loss_dfl(reg_preds_pos, box_targets_pos, anchors_pos, strides_pos, bbox_weight)
        loss_dfl = loss_dfl.sum() / normalizer

        # total loss
        losses = self.loss_cls_weight['tal'] * loss_cls + \
                 self.loss_box_weight['tal'] * loss_box + \
                 self.loss_dfl_weight['tal'] * loss_dfl

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                loss_dfl = loss_dfl,
                losses = losses
        )

        # ------------------ Aux regression loss ------------------
        if epoch >= (self.max_epoch - self.no_aug_epoch - 1) and self.loss_box_aux:
            ## delta_preds
            delta_preds = torch.cat(outputs['pred_delta'], dim=1)
            delta_preds_pos = delta_preds.view(-1, 4)[fg_masks]
            ## aux loss
            loss_box_aux = self.loss_bboxes_aux(delta_preds_pos, box_targets_pos, anchors_pos, strides_pos)
            loss_box_aux = loss_box_aux.sum() / num_fgs

            losses += loss_box_aux
            loss_dict['loss_box_aux'] = loss_box_aux

        return loss_dict
    
    # ----------------- Loss with SimOTA assigner -----------------
    def ota_loss(self, outputs, targets, epoch=0):
        """ Compute loss with SimOTA assigner """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        num_anchors = sum([ab.shape[0] for ab in anchors])
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        reg_preds = torch.cat(outputs['pred_reg'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # --------------- label assignment ---------------
        cls_targets = []
        box_targets = []
        fg_masks = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                # There is no valid gt
                cls_target = cls_preds.new_zeros((num_anchors, self.num_classes))
                box_target = cls_preds.new_zeros((0, 4))
                fg_mask = cls_preds.new_zeros(num_anchors).bool()
            else:
                (
                    fg_mask,
                    assigned_labels,
                    assigned_ious,
                    assigned_indexs
                ) = self.ota_matcher(
                    fpn_strides = fpn_strides,
                    anchors = anchors,
                    pred_cls = cls_preds[batch_idx], 
                    pred_box = box_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )
                # prepare cls targets
                assigned_labels = F.one_hot(assigned_labels.long(), self.num_classes)
                assigned_labels = assigned_labels * assigned_ious.unsqueeze(-1)
                cls_target = assigned_labels.new_zeros((num_anchors, self.num_classes))
                cls_target[fg_mask] = assigned_labels
                # prepare box targets
                box_target = tgt_bboxes[assigned_indexs]

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fgs = fg_masks.sum()

        # average loss normalizer across all the GPUs
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # update loss normalizer with EMA
        if self.use_ema_update:
            normalizer = self.ema_update("loss_normalizer", max(num_fgs, 1), 100)
        else:
            normalizer = num_fgs
        
        # ------------------ Classification loss ------------------
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.loss_classes(cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / normalizer

        # ------------------ Regression loss ------------------
        box_preds_pos = box_preds.view(-1, 4)[fg_masks]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets)
        loss_box = loss_box.sum() / normalizer

        # ------------------ Distribution focal loss  ------------------
        ## process anchors
        anchors = torch.cat(anchors, dim=0)
        anchors = anchors[None].repeat(bs, 1, 1).view(-1, 2)
        ## process stride tensors
        strides = torch.cat(outputs['stride_tensor'], dim=0)
        strides = strides.unsqueeze(0).repeat(bs, 1, 1).view(-1, 1)
        ## fg preds
        reg_preds_pos = reg_preds.view(-1, 4*self.cfg['reg_max'])[fg_masks]
        anchors_pos = anchors[fg_masks]
        strides_pos = strides[fg_masks]
        ## compute dfl
        loss_dfl = self.loss_dfl(reg_preds_pos, box_targets, anchors_pos, strides_pos)
        loss_dfl = loss_dfl.sum() / normalizer

        # total loss
        losses = self.loss_cls_weight['ota'] * loss_cls + \
                 self.loss_box_weight['ota'] * loss_box + \
                 self.loss_dfl_weight['ota'] * loss_dfl

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                loss_dfl = loss_dfl,
                losses = losses
        )

        # ------------------ Aux regression loss ------------------
        if epoch >= (self.max_epoch - self.no_aug_epoch - 1) and self.loss_box_aux:
            ## delta_preds
            delta_preds = torch.cat(outputs['pred_delta'], dim=1)
            delta_preds_pos = delta_preds.view(-1, 4)[fg_masks]
            ## aux loss
            loss_box_aux = self.loss_bboxes_aux(delta_preds_pos, box_targets, anchors_pos, strides_pos)
            loss_box_aux = loss_box_aux.sum() / num_fgs

            losses += loss_box_aux
            loss_dict['loss_box_aux'] = loss_box_aux


        return loss_dict


def build_criterion(args, cfg, device, num_classes):
    criterion = Criterion(
        args=args,
        cfg=cfg,
        device=device,
        num_classes=num_classes
        )

    return criterion


if __name__ == "__main__":
    pass