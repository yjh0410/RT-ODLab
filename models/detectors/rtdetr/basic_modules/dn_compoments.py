import torch


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = torch.split(x, 2, axis=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)

def bbox_xyxy_to_cxcywh(x):
    x1, y1, x2, y2 = x.split(4, axis=-1)
    return torch.cat(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=-1)

def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t) for t in targets["labels"]]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    # pad gt to max_num of a batch
    bs = len(targets["labels"])
    input_query_class = torch.full([bs, max_gt_num], num_classes).long()
    input_query_bbox = torch.zeros([bs, max_gt_num, 4])
    pad_gt_mask = torch.zeros([bs, max_gt_num])
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["labels"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["boxes"][i]
            pad_gt_mask[i, :num_gt] = 1

    # each group has positive and negative queries.
    input_query_class = input_query_class.repeat(1, 2 * num_group)
    input_query_bbox = input_query_bbox.repeat(1, 2 * num_group, 1)
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)

    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1])
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.repeat(1, num_group, 1)
    positive_gt_mask = 1 - negative_gt_mask

    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = torch.rand(input_query_class.shape) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(
            chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class.scatter_(chosen_idx, new_label)
        input_query_class = input_query_class.reshape(bs, num_denoising)
        pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5,
                           [1, 1, 2]) * box_noise_scale

        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand(input_query_bbox.shape)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]])])
    input_query_class = torch.gather(class_embed, 1, input_query_class.flatten())
    input_query_class = input_query_class.reshape(bs, num_denoising, -1)
    
    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones([tgt_size, tgt_size]) < 0
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), max_gt_num *
                      2 * (i + 1):num_denoising] = True
            attn_mask[max_gt_num * 2 * i:max_gt_num * 2 * (i + 1), :max_gt_num *
                      2 * i] = True
    attn_mask = ~attn_mask
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta

