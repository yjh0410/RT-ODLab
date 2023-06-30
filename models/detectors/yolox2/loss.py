import torch
import torch.nn.functional as F
from .matcher import AlignedSimOTA
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
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_box_weight = cfg['loss_box_weight']
        # matcher
        matcher_config = cfg['matcher']
        self.matcher = AlignedSimOTA(
            num_classes=num_classes,
            soft_center_radius=matcher_config['soft_center_radius'],
            topk=matcher_config['topk_candicate'],
            iou_weight=matcher_config['iou_weight']
            )
     
     
    def loss_classes(self, pred_cls, target, beta=2.0):
        """
            Quality Focal Loss
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
        ious = get_ious(pred_box, gt_box, "xyxy", 'giou')
        loss_box = 1.0 - ious

        return loss_box


    def __call__(self, outputs, targets):        
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
        box_preds = torch.cat(outputs['pred_box'], dim=1)

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
        # num_fgs = assign_metrics.sum()
        num_fgs = pos_inds.size(0)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = max(num_fgs / get_world_size(), 1.0)
        
        # cls loss
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self.loss_classes(cls_preds, (cls_targets, assign_metrics))
        loss_cls = loss_cls.sum() / num_fgs

        # regression loss
        box_preds_pos = box_preds.view(-1, 4)[pos_inds]
        box_targets_pos = box_targets[pos_inds]
        loss_box = self.loss_bboxes(box_preds_pos, box_targets_pos)
        loss_box = loss_box.sum() / box_preds_pos.shape[0]

        # total loss
        losses = self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        loss_dict = dict(
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