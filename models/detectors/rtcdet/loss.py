import torch
import torch.nn.functional as F

from utils.box_ops import bbox2dist, get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import build_matcher


# ----------------------- Criterion for training -----------------------
class Criterion(object):
    def __init__(self, args, cfg, device, num_classes=80):
        self.cfg = cfg
        self.args = args
        self.device = device
        self.num_classes = num_classes
        self.max_epoch = args.max_epoch
        self.no_aug_epoch = args.no_aug_epoch
        self.use_ema_update = cfg['ema_update']
        self.loss_box_aux    = cfg['loss_box_aux']
        # ---------------- Loss weight ----------------
        loss_weights = cfg['loss_weights'][cfg['matcher']]
        self.loss_cls_weight = loss_weights['loss_cls_weight']
        self.loss_box_weight = loss_weights['loss_box_weight']
        self.loss_dfl_weight = loss_weights['loss_dfl_weight']
        # ---------------- Matcher ----------------
        ## Aligned SimOTA assigner
        self.matcher = build_matcher(cfg, num_classes)

    def ema_update(self, name: str, value, initial_value, momentum=0.9):
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new

    # ----------------- Loss functions -----------------
    def loss_classes(self, pred_cls, gt_score):
        # compute bce loss
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_score, reduction='none')

        return loss_cls

    def loss_classes_qfl(self, pred_cls, target, beta=2.0):
        # Quality FocalLoss
        """
            pred_cls: (torch.Tensor): [N, C]。
            target:   (tuple([torch.Tensor], [torch.Tensor])): label -> (N,), score -> (N,)
        """
        label, score = target
        pred_sigmoid = pred_cls.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_cls.shape)

        ce_loss = F.binary_cross_entropy_with_logits(
            pred_cls, zerolabel, reduction='none') * scale_factor.pow(beta)
        
        bg_class_ind = pred_cls.shape[-1]
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
        pos_label = label[pos].long()

        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]

        ce_loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred_cls[pos, pos_label], score[pos],
            reduction='none') * scale_factor.abs().pow(beta)

        return ce_loss
    
    def loss_bboxes(self, pred_box, gt_box):
        # regression loss
        ious = get_ious(pred_box, gt_box, 'xyxy', 'giou')
        loss_box = 1.0 - ious

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
    
    # ----------------- Main process -----------------
    def loss_simota(self, outputs, targets, epoch=0):
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
                ) = self.matcher(
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
        losses = self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box + \
                 self.loss_dfl_weight * loss_dfl

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
            loss_box_aux = loss_box_aux.sum() / normalizer

            losses += loss_box_aux
            loss_dict['loss_box_aux'] = loss_box_aux


        return loss_dict

    def loss_aligned_simota(self, outputs, targets, epoch=0):
        """
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        reg_preds = torch.cat(outputs['pred_reg'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # --------------- label assignment ---------------
        cls_targets = []
        box_targets = []
        assign_metrics = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
            # label assignment
            assigned_result = self.matcher(fpn_strides=fpn_strides,
                                           anchors=anchors,
                                           pred_cls=cls_preds[batch_idx].detach(),
                                           pred_box=box_preds[batch_idx].detach(),
                                           gt_labels=tgt_labels,
                                           gt_bboxes=tgt_bboxes
                                           )
            cls_targets.append(assigned_result['assigned_labels'])
            box_targets.append(assigned_result['assigned_bboxes'])
            assign_metrics.append(assigned_result['assign_metrics'])

        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        assign_metrics = torch.cat(assign_metrics, dim=0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((cls_targets >= 0)
                    & (cls_targets < bg_class_ind)).nonzero().squeeze(1)
        num_fgs = assign_metrics.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0).item()
        
        # update loss normalizer with EMA
        if self.use_ema_update:
            normalizer = self.ema_update("loss_normalizer", max(num_fgs, 1), 100)
        else:
            normalizer = num_fgs

        # ---------------------------- Classification loss ----------------------------
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.loss_classes_qfl(cls_preds, (cls_targets, assign_metrics))
        loss_cls = loss_cls.sum() / normalizer

        # ---------------------------- Regression loss ----------------------------
        box_preds_pos = box_preds.view(-1, 4)[pos_inds]
        box_targets_pos = box_targets[pos_inds]
        box_weight_pos = assign_metrics[pos_inds]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets_pos)
        loss_box *= box_weight_pos
        loss_box = loss_box.sum() / normalizer

        # ------------------ Distribution focal loss  ------------------
        ## process anchors
        anchors = torch.cat(anchors, dim=0)
        anchors = anchors[None].repeat(bs, 1, 1).view(-1, 2)
        ## process stride tensors
        strides = torch.cat(outputs['stride_tensor'], dim=0)
        strides = strides.unsqueeze(0).repeat(bs, 1, 1).view(-1, 1)
        ## fg preds
        reg_preds_pos = reg_preds.view(-1, 4*self.cfg['reg_max'])[pos_inds]
        anchors_pos = anchors[pos_inds]
        strides_pos = strides[pos_inds]
        ## compute dfl
        loss_dfl = self.loss_dfl(reg_preds_pos, box_targets_pos, anchors_pos, strides_pos)
        loss_dfl *= box_weight_pos
        loss_dfl = loss_dfl.sum() / normalizer

        # total loss
        losses = self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box + \
                 self.loss_dfl_weight * loss_dfl

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
            delta_preds_pos = delta_preds.view(-1, 4)[pos_inds]
            ## aux loss
            loss_box_aux = self.loss_bboxes_aux(delta_preds_pos, box_targets_pos, anchors_pos, strides_pos)
            loss_box_aux = loss_box_aux.sum() / normalizer

            losses += loss_box_aux
            loss_dict['loss_box_aux'] = loss_box_aux

        return loss_dict


    def __call__(self, outputs, targets, epoch=0):
        if self.cfg['matcher'] == "simota":
            return self.loss_simota(outputs, targets, epoch)
        elif self.cfg['matcher'] == "aligned_simota":
            return self.loss_aligned_simota(outputs, targets, epoch)
        

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