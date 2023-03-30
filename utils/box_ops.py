import torch
import numpy as np
from torchvision.ops.boxes import box_area


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def get_ious(bboxes1,
             bboxes2,
             box_mode="xyxy",
             iou_type="iou"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        bboxes1 = torch.cat((-bboxes1[..., :2], bboxes1[..., 2:]), dim=-1)
        bboxes2 = torch.cat((-bboxes2[..., :2], bboxes2[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]).clamp_(min=0) \
        * (bboxes1[..., 3] - bboxes1[..., 1]).clamp_(min=0)
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]).clamp_(min=0) \
        * (bboxes2[..., 3] - bboxes2[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(bboxes1[..., 2], bboxes2[..., 2])
                   - torch.max(bboxes1[..., 0], bboxes2[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(bboxes1[..., 3], bboxes2[..., 3])
                   - torch.max(bboxes1[..., 1], bboxes2[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = bboxes2_area + bboxes1_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if iou_type == "iou":
        return ious
    elif iou_type == "giou":
        g_w_intersect = torch.max(bboxes1[..., 2], bboxes2[..., 2]) \
            - torch.min(bboxes1[..., 0], bboxes2[..., 0])
        g_h_intersect = torch.max(bboxes1[..., 3], bboxes2[..., 3]) \
            - torch.min(bboxes1[..., 1], bboxes2[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        return gious
    else:
        raise NotImplementedError


def rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas=None):
    origin_h, origin_w = origin_img_size
    cur_img_h, cur_img_w = cur_img_size
    if deltas is None:
        # rescale
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / cur_img_w * origin_w
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / cur_img_h * origin_h

        # clip bboxes
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_w)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_h)
    else:
        # rescale
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / (cur_img_w - deltas[0]) * origin_w
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / (cur_img_h - deltas[1]) * origin_h
        print('deltas')
        
        # clip bboxes
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_w)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_h)

    return bboxes


if __name__ == '__main__':
    box1 = torch.tensor([[10, 10, 20, 20]])
    box2 = torch.tensor([[15, 15, 20, 20]])
    iou = box_iou(box1, box2)
    print(iou)
