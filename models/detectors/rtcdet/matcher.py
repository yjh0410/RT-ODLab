import torch
import torch.nn.functional as F
from utils.box_ops import *


# -------------------------- YOLOX's SimOTA Assigner --------------------------
## Simple OTA
class SimOTA(object):
    """
        This code referenced to https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
    """
    def __init__(self, num_classes, center_sampling_radius, topk_candidate ):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides, 
                 anchors, 
                 pred_cls, 
                 pred_box, 
                 tgt_labels,
                 tgt_bboxes):
        # [M,]
        strides_tensor = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)
        num_anchor = anchors.shape[0]        
        num_gt = len(tgt_labels)

        # ----------------------- Find inside points -----------------------
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            tgt_bboxes, anchors, strides_tensor, num_anchor, num_gt)
        cls_preds = pred_cls[fg_mask].float()   # [Mp, C]
        box_preds = pred_box[fg_mask].float()   # [Mp, 4]

        # ----------------------- Reg cost -----------------------
        pair_wise_ious, _ = box_iou(tgt_bboxes, box_preds)      # [N, Mp]
        reg_cost = -torch.log(pair_wise_ious + 1e-8)            # [N, Mp]

        # ----------------------- Cls cost -----------------------
        with torch.cuda.amp.autocast(enabled=False):
            # [Mp, C] -> [N, Mp, C]
            cls_preds_expand = cls_preds.unsqueeze(0).repeat(num_gt, 1, 1)
            # prepare cls_target
            cls_targets = F.one_hot(tgt_labels.long(), self.num_classes).float()
            cls_targets = cls_targets.unsqueeze(1).repeat(1, cls_preds_expand.size(1), 1)
            cls_targets *= pair_wise_ious.unsqueeze(-1)  # iou-aware
            # [N, Mp]
            cls_cost = F.binary_cross_entropy_with_logits(cls_preds_expand, cls_targets, reduction="none").sum(-1)
        del cls_preds_expand

        #----------------------- Dynamic K-Matching -----------------------
        cost_matrix = (
            cls_cost
            + 3.0 * reg_cost
            + 100000.0 * (~is_in_boxes_and_center)
        ) # [N, Mp]

        (
            assigned_labels,         # [num_fg,]
            assigned_ious,           # [num_fg,]
            assigned_indexs,         # [num_fg,]
        ) = self.dynamic_k_matching(
            cost_matrix,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask
            )
        del cls_cost, cost_matrix, pair_wise_ious, reg_cost

        return fg_mask, assigned_labels, assigned_ious, assigned_indexs


    def get_in_boxes_info(
        self,
        gt_bboxes,   # [N, 4]
        anchors,     # [M, 2]
        strides,     # [M,]
        num_anchors, # M
        num_gt,      # N
        ):
        # anchor center
        x_centers = anchors[:, 0]
        y_centers = anchors[:, 1]

        # [M,] -> [1, M] -> [N, M]
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        # [N,] -> [N, 1] -> [N, M]
        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors) # x1
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors) # y1
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors) # x2
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors) # y2

        b_l = x_centers - gt_bboxes_l
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center
        center_radius = self.center_sampling_radius

        # [N, 2]
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        
        # [1, M]
        center_radius_ = center_radius * strides.unsqueeze(0)

        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # x1
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # y1
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # x2
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # y2

        c_l = x_centers - gt_bboxes_l
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    
    def dynamic_k_matching(
        self, 
        cost, 
        pair_wise_ious, 
        gt_classes, 
        num_gt, 
        fg_mask
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        assigned_indexs = matching_matrix[:, fg_mask_inboxes].argmax(0)
        assigned_labels = gt_classes[assigned_indexs]

        assigned_ious = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return assigned_labels, assigned_ious, assigned_indexs
    

# -------------------------- RTMDet's Aligned SimOTA Assigner --------------------------
## Aligned SimOTA
class AlignedSimOTA(object):
    """
        This code referenced to https://github.com/open-mmlab/mmyolo/models/task_modules/assigners/batch_dsl_assigner.py
    """
    def __init__(self, num_classes, soft_center_radius=3.0, topk=13, iou_weight=3.0):
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides, 
                 anchors, 
                 pred_cls, 
                 pred_box, 
                 gt_labels,
                 gt_bboxes):
        # [M,]
        strides = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)
        num_gt = len(gt_labels)

        # check gt
        if num_gt == 0 or gt_bboxes.max().item() == 0.:
            return {
                'assigned_labels': gt_labels.new_full(pred_cls[..., 0].shape,
                                                      self.num_classes,
                                                      dtype=torch.long),
                'assigned_bboxes': gt_bboxes.new_full(pred_box.shape, 0),
                'assign_metrics': gt_bboxes.new_full(pred_cls[..., 0].shape, 0)
            }
        
        # get inside points: [N, M]
        is_in_gt = self.find_inside_points(gt_bboxes, anchors)
        valid_mask = is_in_gt.sum(dim=0) > 0  # [M,]

        # ----------------------------------- Soft center prior -----------------------------------
        gt_center = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0
        distance = (anchors.unsqueeze(0) - gt_center.unsqueeze(1)
                    ).pow(2).sum(-1).sqrt() / strides.unsqueeze(0)  # [N, M]
        distance = distance * valid_mask.unsqueeze(0)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # ----------------------------------- Regression cost -----------------------------------
        pair_wise_ious, _ = box_iou(gt_bboxes, pred_box)  # [N, M]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) * self.iou_weight

        # ----------------------------------- Classification cost -----------------------------------
        ## select the predicted scores corresponded to the gt_labels
        pairwise_pred_scores = pred_cls.permute(1, 0)  # [M, C] -> [C, M]
        pairwise_pred_scores = pairwise_pred_scores[gt_labels.long(), :].float()   # [N, M]
        ## scale factor
        scale_factor = (pair_wise_ious - pairwise_pred_scores.sigmoid()).abs().pow(2.0)
        ## cls cost
        pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pair_wise_ious,
            reduction="none") * scale_factor # [N, M]
            
        del pairwise_pred_scores

        ## foreground cost matrix
        cost_matrix = pair_wise_cls_loss + pair_wise_ious_loss + soft_center_prior
        max_pad_value = torch.ones_like(cost_matrix) * 1e9
        cost_matrix = torch.where(valid_mask[None].repeat(num_gt, 1),   # [N, M]
                                  cost_matrix, max_pad_value)

        # ----------------------------------- dynamic label assignment -----------------------------------
        (
            matched_pred_ious,
            matched_gt_inds,
            fg_mask_inboxes
        ) = self.dynamic_k_matching(
            cost_matrix,
            pair_wise_ious,
            num_gt
            )
        del pair_wise_cls_loss, cost_matrix, pair_wise_ious, pair_wise_ious_loss

        # -----------------------------------process assigned labels -----------------------------------
        assigned_labels = gt_labels.new_full(pred_cls[..., 0].shape,
                                             self.num_classes)  # [M,]
        assigned_labels[fg_mask_inboxes] = gt_labels[matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()  # [M,]

        assigned_bboxes = gt_bboxes.new_full(pred_box.shape, 0)        # [M, 4]
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[matched_gt_inds]  # [M, 4]

        assign_metrics = gt_bboxes.new_full(pred_cls[..., 0].shape, 0) # [M, 4]
        assign_metrics[fg_mask_inboxes] = matched_pred_ious            # [M, 4]

        assigned_dict = dict(
            assigned_labels=assigned_labels,
            assigned_bboxes=assigned_bboxes,
            assign_metrics=assign_metrics
            )
        
        return assigned_dict


    def find_inside_points(self, gt_bboxes, anchors):
        """
            gt_bboxes: Tensor -> [N, 2]
            anchors:   Tensor -> [M, 2]
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_bboxes.shape[0]

        anchors_expand = anchors.unsqueeze(0).repeat(num_gt, 1, 1)           # [N, M, 2]
        gt_bboxes_expand = gt_bboxes.unsqueeze(1).repeat(1, num_anchors, 1)  # [N, M, 4]

        # offset
        lt = anchors_expand - gt_bboxes_expand[..., :2]
        rb = gt_bboxes_expand[..., 2:] - anchors_expand
        bbox_deltas = torch.cat([lt, rb], dim=-1)

        is_in_gts = bbox_deltas.min(dim=-1).values > 0

        return is_in_gts
    

    def dynamic_k_matching(self, cost_matrix, pairwise_ious, num_gt):
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        # sorting the batch cost matirx is faster than topk
        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for gt_idx in range(num_gt):
            topk_ids = sorted_indices[gt_idx, :dynamic_ks[gt_idx]]
            matching_matrix[gt_idx, :][topk_ids] = 1

        del topk_ious, dynamic_ks, topk_ids

        prior_match_gt_mask = matching_matrix.sum(0) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost_matrix[:, prior_match_gt_mask], dim=0)
            matching_matrix[:, prior_match_gt_mask] *= 0
            matching_matrix[cost_argmin, prior_match_gt_mask] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(0)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes


def build_matcher(cfg, num_classes):
    if cfg['matcher'] == "simota":
        matcher = SimOTA(
            center_sampling_radius=cfg['matcher_hpy'][cfg['matcher']]['center_sampling_radius'],
            topk_candidate=cfg['matcher_hpy'][cfg['matcher']]['topk_candidate'],
            num_classes=num_classes
        )

    elif cfg['matcher'] == "aligned_simota":
        matcher = AlignedSimOTA(
            num_classes=num_classes,
            soft_center_radius=cfg['matcher_hpy'][cfg['matcher']]['soft_center_radius'],
            topk=cfg['matcher_hpy'][cfg['matcher']]['topk_candicate'],
            iou_weight=cfg['matcher_hpy'][cfg['matcher']]['iou_weight']
            )

    return matcher