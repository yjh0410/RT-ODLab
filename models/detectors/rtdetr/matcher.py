import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

try:
    from .loss_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
except:
    from  loss_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class, cost_bbox, cost_giou, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        bs, num_queries = pred_logits.shape[:2]
        # [B, Nq, C] -> [BNq, C]
        out_prob = pred_logits.flatten(0, 1).sigmoid()
        out_bbox = pred_boxes.flatten(0, 1)

        # List[B, M, C] -> [BM, C]
        tgt_ids = torch.cat(gt_labels).long()
        tgt_bbox = torch.cat(gt_boxes).float()

        # -------------------- Classification cost --------------------
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # -------------------- Regression cost --------------------
        ## L1 cost: [Nq, M]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox.to(out_bbox.device), p=1)
        ## GIoU cost: Nq, M]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox).to(out_bbox.device))

        # Final cost: [B, Nq, M]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Label assignment
        sizes = [len(t) for t in gt_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    