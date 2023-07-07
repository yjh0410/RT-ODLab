# ---------------------------------------------------------------------
# Copyright (c) Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------


import torch
import torch.nn.functional as F
from utils.box_ops import *


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
                 pred_obj, 
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
        obj_preds = pred_obj[fg_mask].float()   # [Mp, 1]
        cls_preds = pred_cls[fg_mask].float()   # [Mp, C]
        box_preds = pred_box[fg_mask].float()   # [Mp, 4]

        # ----------------------- Reg cost -----------------------
        pair_wise_ious, _ = box_iou(tgt_bboxes, box_preds)      # [N, Mp]
        reg_cost = -torch.log(pair_wise_ious + 1e-8)            # [N, Mp]

        # ----------------------- Cls cost -----------------------
        with torch.cuda.amp.autocast(enabled=False):
            # [Mp, C]
            score_preds = torch.sqrt(obj_preds.sigmoid_()* cls_preds.sigmoid_())
            # [N, Mp, C]
            score_preds = score_preds.unsqueeze(0).repeat(num_gt, 1, 1)
            # prepare cls_target
            cls_targets = F.one_hot(tgt_labels.long(), self.num_classes).float()
            cls_targets = cls_targets.unsqueeze(1).repeat(1, score_preds.size(1), 1)
            # [N, Mp]
            cls_cost = F.binary_cross_entropy(score_preds, cls_targets, reduction="none").sum(-1)
        del score_preds

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
    