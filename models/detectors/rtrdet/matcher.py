import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"


    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [B * num_queries, C] = [N, C], where N is B * num_queries
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        # [B * num_queries, 4] = [N, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        # [M,] where M is number of all targets in this batch
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # [M, 4] where M is number of all targets in this batch
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        # [N, M]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox.to(out_bbox.device), p=1)

        # Compute the giou cost betwen boxes
        # [N, M]
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox.to(out_bbox.device)))

        # Final cost matrix: [N, M]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # [N, M] -> [B, num_queries, M]
        C = C.view(bs, num_queries, -1).cpu()

        # The number of boxes in each image
        sizes = [len(v["boxes"]) for v in targets]
        # In the last dimension of C, we divide it into B costs, and each cost is [B, num_querys, M_i]
        # where sum(Mi) = M.
        # i is the batch index and c is cost_i = [B, num_querys, M_i].
        # Therefore c[i] is the cost between the i-th sample and i-th prediction.
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # As for each (i, j) in indices, i is the prediction indexes and j is the target indexes
        # i contains row indexes of cost matrix: array([row_1, row_2, row_3]) 
        # j contains col indexes of cost matrix: array([col_1, col_2, col_3])
        # len(i) == len(j)
        # len(indices) = batch_size
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    matcher_type = cfg['matcher']
    if matcher_type == 'hungarian_matcher':
        return HungarianMatcher(cfg['matcher_hpy'][matcher_type]['cost_cls_weight'],
                                cfg['matcher_hpy'][matcher_type]['cost_box_weight'],
                                cfg['matcher_hpy'][matcher_type]['cost_giou_weight'])
