import torch


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

def bbox_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def bbox_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t["labels"]) for t in targets]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    # pad gt to max_num of a batch
    bs = len(targets)
    # [bs, max_gt_num]
    input_query_class = torch.full([bs, max_gt_num], num_classes, device=class_embed.device).long()
    # [bs, max_gt_num, 4]
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=class_embed.device)
    # [bs, max_gt_num]
    pad_gt_mask = torch.zeros([bs, max_gt_num], device=class_embed.device)
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1

    # each group has positive and negative queries.
    input_query_class = input_query_class.repeat(1, 2 * num_group)  # [bs, 2*num_denoising], num_denoising = 2 * num_group * max_gt_num
    input_query_bbox = input_query_bbox.repeat(1, 2 * num_group, 1) # [bs, 2*num_denoising, 4]
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)              # [bs, 2*num_denoising]

    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=class_embed.device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.repeat(1, num_group, 1)
    positive_gt_mask = 1 - negative_gt_mask

    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)  # num_denoising *= 2

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()  # [bs * num_denoising]
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = torch.rand(input_query_class.shape, device=class_embed.device) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(
            chosen_idx, 0, num_classes, dtype=input_query_class.dtype, device=class_embed.device) # [b * num_denoising]
        # [bs * num_denoising]
        input_query_class = torch.scatter(input_query_class, 0, chosen_idx, new_label)
        # input_query_class.scatter_(chosen_idx, new_label)
        # [bs * num_denoising] -> # [bs, num_denoising]
        input_query_class = input_query_class.reshape(bs, num_denoising)
        pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = bbox_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5,
                           [1, 1, 2]) * box_noise_scale

        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand(input_query_bbox.shape, device=class_embed.device)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clamp_(min=0.0, max=1.0)
        input_query_bbox = bbox_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # [num_classes + 1, hidden_dim]
    class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
    # input_query_class = paddle.gather(class_embed, input_query_class.flatten(), axis=0)

    # input_query_class: [bs, num_denoising] -> [bs*num_denoising, hidden_dim]
    input_query_class = torch.torch.index_select(class_embed, 0, input_query_class.flatten())
    # [bs*num_denoising, hidden_dim] -> [bs, num_denoising, hidden_dim]
    input_query_class = input_query_class.reshape(bs, num_denoising, -1)
    
    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones([tgt_size, tgt_size], device=class_embed.device) < 0
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

