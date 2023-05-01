import torch
import numpy as np


class Yolov2Matcher(object):
    def __init__(self, iou_thresh, num_classes, anchor_size):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        # anchor box
        self.num_anchors = len(anchor_size)
        self.anchor_size = anchor_size
        self.anchor_boxes = np.array(
            [[0., 0., anchor[0], anchor[1]]
            for anchor in anchor_size]
            )  # [KA, 4]


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


    @torch.no_grad()
    def __call__(self, fmp_size, stride, targets):
        """
            img_size: (Int) input image size
            stride: (Int) -> stride of YOLOv1 output.
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        # prepare
        bs = len(targets)
        fmp_h, fmp_w = fmp_size
        gt_objectness = np.zeros([bs, fmp_h, fmp_w, self.num_anchors, 1]) 
        gt_classes = np.zeros([bs, fmp_h, fmp_w, self.num_anchors, self.num_classes]) 
        gt_bboxes = np.zeros([bs, fmp_h, fmp_w, self.num_anchors, 4])

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                x1, y1, x2, y2 = gt_box
                # xyxy -> cxcywh
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                gt_box = [0, 0, bw, bh]

                # check
                if bw < 1. or bh < 1.:
                    continue    

                # compute IoU
                iou = self.compute_iou(self.anchor_boxes, gt_box)
                iou_mask = (iou > self.iou_thresh)

                label_assignment_results = []
                if iou_mask.sum() == 0:
                    # We assign the anchor box with highest IoU score.
                    iou_ind = np.argmax(iou)
                    anchor_idx = iou_ind
                    # compute the grid cell
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, anchor_idx])
                else:            
                    for iou_ind, iou_m in enumerate(iou_mask):
                        if iou_m:
                            anchor_idx = iou_ind
                            # compute the gride cell
                            xc_s = xc / stride
                            yc_s = yc / stride
                            grid_x = int(xc_s)
                            grid_y = int(yc_s)

                            label_assignment_results.append([grid_x, grid_y, anchor_idx])

                # label assignment
                for result in label_assignment_results:
                    grid_x, grid_y, anchor_idx = result
                    if grid_x < fmp_w and grid_y < fmp_h:
                        # obj
                        gt_objectness[batch_index, grid_y, grid_x, anchor_idx] = 1.0
                        # cls
                        cls_ont_hot = np.zeros(self.num_classes)
                        cls_ont_hot[int(gt_label)] = 1.0
                        gt_classes[batch_index, grid_y, grid_x, anchor_idx] = cls_ont_hot
                        # box
                        gt_bboxes[batch_index, grid_y, grid_x, anchor_idx] = np.array([x1, y1, x2, y2])

        # [B, H, W, A, C] -> [B, HWA, C]
        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_classes = gt_classes.reshape(bs, -1, self.num_classes)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)

        # to tensor
        gt_objectness = torch.from_numpy(gt_objectness).float()
        gt_classes = torch.from_numpy(gt_classes).float()
        gt_bboxes = torch.from_numpy(gt_bboxes).float()

        return gt_objectness, gt_classes, gt_bboxes
