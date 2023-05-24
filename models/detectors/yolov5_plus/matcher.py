import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import bbox_iou


# -------------------------- YOLOv5 Assigner --------------------------
class Yolov5Matcher(object):
    def __init__(self, num_classes, num_anchors, anchor_size, anchor_theshold):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_theshold = anchor_theshold
        # [KA, 2]
        self.anchor_sizes = np.array([[anchor[0], anchor[1]]
                                      for anchor in anchor_size])
        # [KA, 4]
        self.anchor_boxes = np.array([[0., 0., anchor[0], anchor[1]]
                                      for anchor in anchor_size])

    def compute_iou(self, anchor_boxes, gt_box):
        """
            anchor_boxes : ndarray -> [KA, 4] (cx, cy, bw, bh).
            gt_box : ndarray -> [1, 4] (cx, cy, bw, bh).
        """
        # anchors: [KA, 4]
        anchors = np.zeros_like(anchor_boxes)
        anchors[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5  # x1y1
        anchors[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5  # x2y2
        anchors_area = anchor_boxes[..., 2] * anchor_boxes[..., 3]
        
        # gt_box: [1, 4] -> [KA, 4]
        gt_box = np.array(gt_box).reshape(-1, 4)
        gt_box = np.repeat(gt_box, anchors.shape[0], axis=0)
        gt_box_ = np.zeros_like(gt_box)
        gt_box_[..., :2] = gt_box[..., :2] - gt_box[..., 2:] * 0.5  # x1y1
        gt_box_[..., 2:] = gt_box[..., :2] + gt_box[..., 2:] * 0.5  # x2y2
        gt_box_area = np.prod(gt_box[..., 2:] - gt_box[..., :2], axis=1)

        # intersection
        inter_w = np.minimum(anchors[:, 2], gt_box_[:, 2]) - \
                  np.maximum(anchors[:, 0], gt_box_[:, 0])
        inter_h = np.minimum(anchors[:, 3], gt_box_[:, 3]) - \
                  np.maximum(anchors[:, 1], gt_box_[:, 1])
        inter_area = inter_w * inter_h
        
        # union
        union_area = anchors_area + gt_box_area - inter_area

        # iou
        iou = inter_area / union_area
        iou = np.clip(iou, a_min=1e-10, a_max=1.0)
        
        return iou


    def iou_assignment(self, ctr_points, gt_box, fpn_strides):
        # compute IoU
        iou = self.compute_iou(self.anchor_boxes, gt_box)
        iou_mask = (iou > 0.5)

        label_assignment_results = []
        if iou_mask.sum() == 0:
            # We assign the anchor box with highest IoU score.
            iou_ind = np.argmax(iou)

            level = iou_ind // self.num_anchors              # pyramid level
            anchor_idx = iou_ind - level * self.num_anchors  # anchor index

            # get the corresponding stride
            stride = fpn_strides[level]

            # compute the grid cell
            xc, yc = ctr_points
            xc_s = xc / stride
            yc_s = yc / stride
            grid_x = int(xc_s)
            grid_y = int(yc_s)

            label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])
        else:            
            for iou_ind, iou_m in enumerate(iou_mask):
                if iou_m:
                    level = iou_ind // self.num_anchors              # pyramid level
                    anchor_idx = iou_ind - level * self.num_anchors  # anchor index

                    # get the corresponding stride
                    stride = fpn_strides[level]

                    # compute the gride cell
                    xc, yc = ctr_points
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])

        return label_assignment_results


    def aspect_ratio_assignment(self, ctr_points, keeps, fpn_strides):
        label_assignment_results = []
        for keep_idx, keep in enumerate(keeps):
            if keep:
                level = keep_idx // self.num_anchors              # pyramid level
                anchor_idx = keep_idx - level * self.num_anchors  # anchor index

                # get the corresponding stride
                stride = fpn_strides[level]

                # compute the gride cell
                xc, yc = ctr_points
                xc_s = xc / stride
                yc_s = yc / stride
                grid_x = int(xc_s)
                grid_y = int(yc_s)

                label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])
        
        return label_assignment_results
    

    @torch.no_grad()
    def __call__(self, fmp_sizes, fpn_strides, targets):
        """
            fmp_size: (List) [fmp_h, fmp_w]
            fpn_strides: (List) -> [8, 16, 32, ...] stride of network output.
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        assert len(fmp_sizes) == len(fpn_strides)
        # prepare
        bs = len(targets)
        gt_objectness = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 1]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]
        gt_classes = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, self.num_classes]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]
        gt_bboxes = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 4]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # get a bbox coords
                x1, y1, x2, y2 = gt_box.tolist()
                # xyxy -> cxcywh
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                gt_box = np.array([[0., 0., bw, bh]])

                # check target
                if bw < 1. or bh < 1.:
                    # invalid target
                    continue

                # compute aspect ratio
                ratios = gt_box[..., 2:] / self.anchor_sizes
                keeps = np.maximum(ratios, 1 / ratios).max(-1) < self.anchor_theshold

                if keeps.sum() == 0:
                    label_assignment_results = self.iou_assignment([xc, yc], gt_box, fpn_strides)
                else:
                    label_assignment_results = self.aspect_ratio_assignment([xc, yc], keeps, fpn_strides)

                # label assignment
                for result in label_assignment_results:
                    # assignment
                    grid_x, grid_y, xc_s, yc_s, level, anchor_idx = result
                    stride = fpn_strides[level]
                    fmp_h, fmp_w = fmp_sizes[level]
                    # coord on the feature
                    x1s, y1s = x1 / stride, y1 / stride
                    x2s, y2s = x2 / stride, y2 / stride
                    # offset
                    off_x = xc_s - grid_x
                    off_y = yc_s - grid_y
 
                    if off_x <= 0.5 and off_y <= 0.5:  # top left
                        grids = [(grid_x-1, grid_y), (grid_x, grid_y-1), (grid_x, grid_y)]
                    elif off_x > 0.5 and off_y <= 0.5: # top right
                        grids = [(grid_x+1, grid_y), (grid_x, grid_y-1), (grid_x, grid_y)]
                    elif off_x <= 0.5 and off_y > 0.5: # bottom left
                        grids = [(grid_x-1, grid_y), (grid_x, grid_y+1), (grid_x, grid_y)]
                    elif off_x > 0.5 and off_y > 0.5:  # bottom right
                        grids = [(grid_x+1, grid_y), (grid_x, grid_y+1), (grid_x, grid_y)]

                    for (i, j) in grids:
                        is_in_box = (j >= y1s and j < y2s) and (i >= x1s and i < x2s)
                        is_valid = (j >= 0 and j < fmp_h) and (i >= 0 and i < fmp_w)

                        if is_in_box and is_valid:
                            # obj
                            gt_objectness[level][batch_index, j, i, anchor_idx] = 1.0
                            # cls
                            cls_ont_hot = torch.zeros(self.num_classes)
                            cls_ont_hot[int(gt_label)] = 1.0
                            gt_classes[level][batch_index, j, i, anchor_idx] = cls_ont_hot
                            # box
                            gt_bboxes[level][batch_index, j, i, anchor_idx] = torch.as_tensor([x1, y1, x2, y2])

        # [B, M, C]
        gt_objectness = torch.cat([gt.view(bs, -1, 1) for gt in gt_objectness], dim=1).float()
        gt_classes = torch.cat([gt.view(bs, -1, self.num_classes) for gt in gt_classes], dim=1).float()
        gt_bboxes = torch.cat([gt.view(bs, -1, 4) for gt in gt_bboxes], dim=1).float()

        return gt_objectness, gt_classes, gt_bboxes


# -------------------------- Task Aligned Assigner --------------------------
class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk=10,
                 num_classes=80,
                 alpha=0.5,
                 beta=6.0, 
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points):
        # get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (b, max_num_obj, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts

        return mask_pos, align_metric, overlaps


    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False,
                            CIoU=True).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps


    def select_topk_candidates(self, metrics, largest=True):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs[~topk_mask] = 0
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)


    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """

        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
    

# -------------------------- Basic Functions --------------------------
def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt
    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.
    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)
    target_gt_idx = mask_pos.argmax(axis=-2)
    return target_gt_idx, fg_mask , mask_pos


def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch
    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union
