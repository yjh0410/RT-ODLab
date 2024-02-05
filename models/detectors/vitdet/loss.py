import torch
import torch.nn.functional as F

try:
    from .loss_utils import get_ious, get_world_size, is_dist_avail_and_initialized
    from .matcher import AlignedSimOtaMatcher
except:
    from  loss_utils import get_ious, get_world_size, is_dist_avail_and_initialized
    from  matcher import AlignedSimOtaMatcher


class Criterion(object):
    def __init__(self, cfg, num_classes=80):
        # ------------ Basic parameters ------------
        self.cfg = cfg
        self.num_classes = num_classes
        # --------------- Matcher config ---------------
        self.matcher_hpy = cfg['matcher_hpy']
        self.matcher = AlignedSimOtaMatcher(soft_center_radius = self.matcher_hpy['soft_center_radius'],
                                            topk_candidates    = self.matcher_hpy['topk_candidates'],
                                            num_classes        = num_classes,
                                            )
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls':  cfg['loss_coeff']['class'],
                            'loss_box':  cfg['loss_coeff']['bbox'],
                            'loss_giou': cfg['loss_coeff']['giou']}

    def loss_classes(self, pred_cls, target, num_gts, beta=2.0):
        # Quality FocalLoss
        """
            pred_cls: (torch.Tensor): [N, C]。
            target:   (tuple([torch.Tensor], [torch.Tensor])): label -> (N,), score -> (N)
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
        
        losses = {}
        losses['loss_cls'] = ce_loss.sum() / num_gts

        return losses
    
    def loss_bboxes(self, pred_reg, pred_box, gt_box, anchors, stride_tensors, num_gts):
        # --------------- Compute L1 loss ---------------
        ## xyxy -> cxcy&bwbh
        gt_cxcy = (gt_box[..., :2] + gt_box[..., 2:]) * 0.5
        gt_bwbh = gt_box[..., 2:] - gt_box[..., :2]
        ## Encode gt box
        gt_cxcy_encode = (gt_cxcy - anchors) / stride_tensors
        gt_bwbh_encode = torch.log(gt_bwbh / stride_tensors)
        gt_box_encode = torch.cat([gt_cxcy_encode, gt_bwbh_encode], dim=-1)
        # L1 loss
        loss_box = F.l1_loss(pred_reg, gt_box_encode, reduction='none')

        # --------------- Compute GIoU loss ---------------
        gious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type='giou')
        loss_giou = 1.0 - gious

        losses = {}
        losses['loss_box'] = loss_box.sum() / num_gts
        losses['loss_giou'] = loss_giou.sum() / num_gts

        return losses
    
    def __call__(self, outputs, targets):        
        """
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        anchors = outputs['anchors']
        fpn_strides = outputs['strides']
        stride_tensors = outputs['stride_tensors']
        losses = dict()
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        reg_preds = torch.cat(outputs['pred_reg'], dim=1)
        
        # --------------- label assignment ---------------
        cls_targets = []
        box_targets = []
        assign_metrics = []
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N,]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]
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

        # List[B, M, C] -> Tensor[BM, C]
        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        assign_metrics = torch.cat(assign_metrics, dim=0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((cls_targets >= 0) & (cls_targets < bg_class_ind)).nonzero().squeeze(1)
        num_fgs = assign_metrics.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0).item()

        # ------------------ Classification loss ------------------
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_dict = self.loss_classes(cls_preds, (cls_targets, assign_metrics), num_fgs)
        loss_dict = {k: loss_dict[k] * self.weight_dict[k] for k in loss_dict if k in self.weight_dict}
        losses.update(loss_dict)

        # ------------------ Regression loss ------------------
        box_targets_pos = box_targets[pos_inds]
        ## positive predictions
        box_preds_pos = box_preds.view(-1, 4)[pos_inds]
        reg_preds_pos = reg_preds.view(-1, 4)[pos_inds]

        ## anchor tensors
        anchors_tensors = torch.cat(anchors, dim=0)[None].repeat(bs, 1, 1)
        anchors_tensors_pos = anchors_tensors.view(-1, 2)[pos_inds]

        ## stride tensors
        stride_tensors = torch.cat(stride_tensors, dim=0)[None].repeat(bs, 1, 1)
        stride_tensors_pos = stride_tensors.view(-1, 1)[pos_inds]

        ## aux loss
        loss_dict = self.loss_bboxes(reg_preds_pos, box_preds_pos, box_targets_pos, anchors_tensors_pos, stride_tensors_pos, num_fgs)
        loss_dict = {k: loss_dict[k] * self.weight_dict[k] for k in loss_dict if k in self.weight_dict}
        losses.update(loss_dict)

        return losses
    

def build_criterion(cfg, num_classes):
    criterion = Criterion(cfg, num_classes)

    return criterion
