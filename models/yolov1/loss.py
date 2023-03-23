import torch
import torch.nn.functional as F
from .matcher import YoloMatcher


class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = cfg['loss_obj_weight']
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_txty_weight = cfg['loss_txty_weight']
        self.loss_twth_weight = cfg['loss_twth_weight']

        # matcher
        self.matcher = YoloMatcher(num_classes=num_classes)


    def loss_objectness(self, pred_obj, gt_obj):
        obj_score = torch.clamp(torch.sigmoid(pred_obj), min=1e-4, max=1.0 - 1e-4)
        # obj loss
        pos_id = (gt_obj==1.0).float()
        pos_loss = pos_id * (obj_score - gt_obj)**2

        # noobj loss
        neg_id = (gt_obj==0.0).float()
        neg_loss = neg_id * (obj_score)**2

        # total loss
        loss_obj = 5.0 * pos_loss + 1.0 * neg_loss

        return loss_obj
    

    def loss_labels(self, pred_cls, gt_label):
        loss_cls = F.cross_entropy(pred_cls, gt_label, reduction='none')

        return loss_cls


    def loss_txty(self, pred_txty, gt_txty, gt_box_weight):
        # txty loss
        loss_txty = F.binary_cross_entropy_with_logits(
            pred_txty, gt_txty, reduction='none').sum(-1)
        loss_txty *= gt_box_weight

        return loss_txty


    def loss_twth(self, pred_twth, gt_twth, gt_box_weight):
        # twth loss
        loss_twth = F.mse_loss(pred_twth, gt_twth, reduction='none').sum(-1)
        loss_twth *= gt_box_weight

        return loss_twth


    def __call__(self, outputs, targets):
        device = outputs['pred_cls'][0].device
        stride = outputs['stride']
        img_size = outputs['img_size']
        (
            gt_objectness, 
            gt_labels, 
            gt_bboxes,
            gt_box_weight
            ) = self.matcher(img_size=img_size, 
                             stride=stride, 
                             targets=targets)
        # List[B, M, C] -> [B, M, C] -> [BM, C]
        batch_size = outputs['pred_obj'].shape[0]
        pred_obj = outputs['pred_obj'].view(-1)                     # [BM,]
        pred_cls = outputs['pred_cls'].view(-1, self.num_classes)   # [BM, C]
        pred_txty = outputs['pred_txty'].view(-1, 2)                # [BM, 2]
        pred_twth = outputs['pred_twth'].view(-1, 2)                # [BM, 2]
       
        gt_objectness = gt_objectness.view(-1).to(device).float()   # [BM,]
        gt_labels = gt_labels.view(-1).to(device).long()            # [BM,]
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()        # [BM, 4]
        gt_box_weight = gt_box_weight.view(-1).to(device).float()   # [BM,]

        pos_masks = (gt_objectness > 0)

        # objectness loss
        loss_obj = self.loss_objectness(pred_obj, gt_objectness)
        loss_obj = loss_obj.sum() / batch_size

        # classification loss
        pred_cls_pos = pred_cls[pos_masks]
        gt_labels_pos = gt_labels[pos_masks]
        loss_cls = self.loss_labels(pred_cls_pos, gt_labels_pos)
        loss_cls = loss_cls.sum() / batch_size

        # txty loss
        pred_txty_pos = pred_txty[pos_masks]
        gt_txty_pos = gt_bboxes[pos_masks][..., :2]
        gt_box_weight_pos = gt_box_weight[pos_masks]
        loss_txty = self.loss_txty(pred_txty_pos, gt_txty_pos, gt_box_weight_pos)
        loss_txty = loss_txty.sum() / batch_size
        
        # twth loss
        pred_twth_pos = pred_twth[pos_masks]
        gt_twth_pos = gt_bboxes[pos_masks][..., 2:]
        loss_twth = self.loss_twth(pred_twth_pos, gt_twth_pos, gt_box_weight_pos)
        loss_twth = loss_twth.sum() / batch_size

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_txty_weight * loss_txty + \
                 self.loss_twth_weight * loss_twth

        loss_dict = dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_txty = loss_txty,
                loss_twth = loss_twth,
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
