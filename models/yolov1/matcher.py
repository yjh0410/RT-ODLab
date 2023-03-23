import torch
import numpy as np


class YoloMatcher(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes


    def generate_dxdywh(self, gt_box, img_size, stride):
        x1, y1, x2, y2 = gt_box
        # xyxy -> cxcywh
        xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
        bw, bh = x2 - x1, y2 - y1

        # 检查数据的有效性
        if bw < 1. or bh < 1.:
            return False    

        # 计算中心点所在的网格坐标
        xs_c = xc / stride
        ys_c = yc / stride
        grid_x = int(xs_c)
        grid_y = int(ys_c)

        # 计算中心点偏移量和宽高的标签
        tx = xs_c - grid_x
        ty = ys_c - grid_y
        tw = np.log(bw)
        th = np.log(bh)

        # 计算边界框位置参数的损失权重
        weight = 2.0 - (bh / img_size[0]) * (bw / img_size[1])

        return grid_x, grid_y, tx, ty, tw, th, weight


    @torch.no_grad()
    def __call__(self, img_size, stride, targets):
        """
            img_size: (Int) input image size
            stride: (Int) -> stride of YOLOv1 output.
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        # prepare
        bs = len(targets)
        fmp_h, fmp_w = img_size[0] // stride, img_size[1] // stride
        gt_objectness = np.zeros([bs, fmp_h, fmp_w, 1]) 
        gt_labels = np.zeros([bs, fmp_h, fmp_w, 1]) 
        gt_bboxes = np.zeros([bs, fmp_h, fmp_w, 4])
        gt_box_weight = np.zeros([bs, fmp_h, fmp_w, 1])

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                result = self.generate_dxdywh(gt_box, img_size, stride)
                if result:
                    grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < fmp_w and grid_y < fmp_h:
                    gt_objectness[batch_index, grid_y, grid_x] = 1.0
                    gt_labels[batch_index, grid_y, grid_x] = gt_label
                    gt_bboxes[batch_index, grid_y, grid_x] = np.array([tx, ty, tw, th])
                    gt_box_weight[batch_index, grid_y, grid_x] = weight

        # [B, M, C]
        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_labels = gt_labels.reshape(bs, -1, 1)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)
        gt_box_weight = gt_box_weight.reshape(bs, -1, 1)

        # to tensor
        gt_objectness = torch.from_numpy(gt_objectness).float()
        gt_labels = torch.from_numpy(gt_labels).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).float()
        gt_box_weight = torch.from_numpy(gt_box_weight).float()

        return gt_objectness, gt_labels, gt_bboxes, gt_box_weight
