import torch
import torch.nn.functional as F

from utils.box_ops import bbox2dist, get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import AlignedSimOTA


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
        self.loss_box_aux    = cfg['loss_box_aux']
        # ---------------- Matcher ----------------
        matcher_config = cfg['matcher']
        ## Aligned SimOTA assigner
        self.ota_matcher = AlignedSimOTA(
            num_classes=num_classes,
            soft_center_radius=matcher_config['soft_center_radius'],
            topk_candidate=matcher_config['topk_candidate'],
            iou_weight=matcher_config['iou_weight']
        )


    def ema_update(self, name: str, value, initial_value, momentum=0.9):
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new


    def loss_classes(self, pred_cls, target, beta=2.0):
        # Quality FocalLoss
        """
            pred_cls: (torch.Tensor): [N, C]。
            target:   (tuple([torch.Tensor], [torch.Tensor])): label -> [N,], score -> [N,]
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


    def loss_bboxes_aux(self, pred_reg, gt_box, anchors, stride_tensors):
        # xyxy -> cxcy&bwbh
        gt_cxcy = (gt_box[..., :2] + gt_box[..., 2:]) * 0.5
        gt_bwbh = gt_box[..., 2:] - gt_box[..., :2]
        # encode gt box
        gt_cxcy_encode = (gt_cxcy - anchors) / stride_tensors
        gt_bwbh_encode = torch.log(gt_bwbh / stride_tensors)
        gt_box_encode = torch.cat([gt_cxcy_encode, gt_bwbh_encode], dim=-1)
        # l1 loss
        loss_box_aux = F.l1_loss(pred_reg, gt_box_encode, reduction='none')

        return loss_box_aux


    def __call__(self, outputs, targets, epoch=0):
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
        assign_metrics = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
            # label assignment
            assigned_result = self.ota_matcher(fpn_strides=fpn_strides,
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
        pos_inds = ((cls_targets >= 0) & (cls_targets < bg_class_ind)).nonzero().squeeze(1)
        num_fgs = assign_metrics.sum()
        
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
        loss_cls = self.loss_classes(cls_preds, (cls_targets, assign_metrics))
        loss_cls = loss_cls.sum() / normalizer

        # ------------------ Regression loss ------------------
        box_preds_pos = box_preds.view(-1, 4)[pos_inds]
        box_targets_pos = box_targets[pos_inds]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets_pos)
        loss_box = loss_box.sum() / normalizer

        losses = self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        loss_dict = dict(
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        # ------------------ Aux regression loss ------------------
        if epoch >= (self.max_epoch - self.no_aug_epoch - 1):
            ## reg_preds
            reg_preds = torch.cat(outputs['pred_reg'], dim=1)
            reg_preds_pos = reg_preds.view(-1, 4)[pos_inds]
            ## anchor tensors
            anchors_tensors = torch.cat(outputs['anchors'], dim=0)[None].repeat(bs, 1, 1)
            anchors_tensors_pos = anchors_tensors.view(-1, 2)[pos_inds]
            ## stride tensors
            stride_tensors = torch.cat(outputs['stride_tensors'], dim=0)[None].repeat(bs, 1, 1)
            stride_tensors_pos = stride_tensors.view(-1, 1)[pos_inds]
            ## aux loss
            loss_box_aux = self.loss_bboxes_aux(reg_preds_pos, box_targets_pos, anchors_tensors_pos, stride_tensors_pos)
            loss_box_aux = loss_box_aux.sum() / normalizer

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