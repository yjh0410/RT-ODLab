import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .loss_utils import varifocal_loss_with_logits, sigmoid_focal_loss
    from .loss_utils import box_cxcywh_to_xyxy, bbox_iou
    from .loss_utils import is_dist_avail_and_initialized, get_world_size
    from .loss_utils import GIoULoss
    from .matcher import HungarianMatcher
except:
    from loss_utils import varifocal_loss_with_logits, sigmoid_focal_loss
    from loss_utils import box_cxcywh_to_xyxy, bbox_iou
    from loss_utils import is_dist_avail_and_initialized, get_world_size
    from loss_utils import GIoULoss
    from matcher import HungarianMatcher


# --------------- Criterion for RT-DETR ---------------
def build_criterion(cfg, num_classes=80):
    return Criterion(cfg, num_classes)

class Criterion(object):
    def __init__(self, cfg, num_classes=80):
        self.matcher = HungarianMatcher(cfg['matcher_hpy']['cost_class'],
                                        cfg['matcher_hpy']['cost_bbox'],
                                        cfg['matcher_hpy']['cost_giou'],
                                        alpha=0.25,
                                        gamma=2.0)
        self.loss = DINOLoss(num_classes   = num_classes,
                                matcher    = self.matcher,
                                aux_loss   = True,
                                use_vfl    = cfg['use_vfl'],
                                loss_coeff = cfg['loss_coeff'])

    def __call__(self, dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta, targets=None):
        assert targets is not None

        gt_labels = [t['labels'].to(dec_out_bboxes.device) for t in targets]  # (List[torch.Tensor]) -> List[[N,]]
        gt_boxes  = [t['boxes'].to(dec_out_bboxes.device)  for t in targets]  # (List[torch.Tensor]) -> List[[N, 4]]

        if dn_meta is not None:
            if isinstance(dn_meta, list):
                dual_groups = len(dn_meta) - 1
                dec_out_bboxes = torch.chunk(
                    dec_out_bboxes, dual_groups + 1, dim=2)
                dec_out_logits = torch.chunk(
                    dec_out_logits, dual_groups + 1, dim=2)
                enc_topk_bboxes = torch.chunk(
                    enc_topk_bboxes, dual_groups + 1, dim=1)
                enc_topk_logits = torch.splchunkt(
                    enc_topk_logits, dual_groups + 1, dim=1)

                loss = {}
                for g_id in range(dual_groups + 1):
                    if dn_meta[g_id] is not None:
                        dn_out_bboxes_gid, dec_out_bboxes_gid = torch.split(
                            dec_out_bboxes[g_id],
                            dn_meta[g_id]['dn_num_split'],
                            dim=2)
                        dn_out_logits_gid, dec_out_logits_gid = torch.split(
                            dec_out_logits[g_id],
                            dn_meta[g_id]['dn_num_split'],
                            dim=2)
                    else:
                        dn_out_bboxes_gid, dn_out_logits_gid = None, None
                        dec_out_bboxes_gid = dec_out_bboxes[g_id]
                        dec_out_logits_gid = dec_out_logits[g_id]
                    out_bboxes_gid = torch.cat([
                        enc_topk_bboxes[g_id].unsqueeze(0),
                        dec_out_bboxes_gid
                    ])
                    out_logits_gid = torch.cat([
                        enc_topk_logits[g_id].unsqueeze(0),
                        dec_out_logits_gid
                    ])
                    loss_gid = self.loss(
                        out_bboxes_gid,
                        out_logits_gid,
                        gt_boxes,
                        gt_labels,
                        dn_out_bboxes=dn_out_bboxes_gid,
                        dn_out_logits=dn_out_logits_gid,
                        dn_meta=dn_meta[g_id])
                    # sum loss
                    for key, value in loss_gid.items():
                        loss.update({
                            key: loss.get(key, torch.zeros([1], device=out_bboxes_gid.device)) + value
                        })

                # average across (dual_groups + 1)
                for key, value in loss.items():
                    loss.update({key: value / (dual_groups + 1)})
                return loss
            else:
                dn_out_bboxes, dec_out_bboxes = torch.split(
                    dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
                dn_out_logits, dec_out_logits = torch.split(
                    dec_out_logits, dn_meta['dn_num_split'], dim=2)
        else:
            dn_out_bboxes, dn_out_logits = None, None

        out_bboxes = torch.cat(
            [enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
        out_logits = torch.cat(
            [enc_topk_logits.unsqueeze(0), dec_out_logits])

        return self.loss(out_bboxes,
                         out_logits,
                         gt_boxes,
                         gt_labels,
                         dn_out_bboxes=dn_out_bboxes,
                         dn_out_logits=dn_out_logits,
                         dn_meta=dn_meta)


# --------------- DETR series loss ---------------
class DETRLoss(nn.Module):
    """Modified Paddle DETRLoss class without mask loss."""
    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 aux_loss=True,
                 use_vfl=False,
                 loss_coeff={'class': 1,
                             'bbox':  5,
                             'giou':  2,},
                 ):
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_vfl = use_vfl
        self.giou_loss = GIoULoss(reduction='none')

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = torch.full(logits.shape[:2], bg_index, device=logits.device).long()
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(
                num_query_objects, gt_class, match_indices)
            target_label = target_label.reshape(-1, 1)
            target_label[index] = updates.long()[:, None]
            # target_label = paddle.scatter(target_label, index, updates.long())
            target_label = target_label.reshape(bs, num_query_objects)

        # one-hot label
        target_label = F.one_hot(target_label, self.num_classes + 1)[..., :-1].float()
        if iou_score is not None and self.use_vfl:
            target_score = torch.zeros([bs, num_query_objects], device=logits.device)
            if num_gt > 0:
                target_score = target_score.reshape(-1, 1)
                target_score[index] = iou_score.float()
                # target_score = paddle.scatter(target_score, index, iou_score)
            target_score = target_score.reshape(bs, num_query_objects, 1) * target_label
            loss_cls = varifocal_loss_with_logits(logits,
                                                  target_score,
                                                  target_label,
                                                  num_gts)
        else:
            loss_cls = sigmoid_focal_loss(logits,
                                          target_label,
                                          num_gts)

        return {name_class: loss_cls * self.loss_coeff['class']}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = torch.as_tensor([0.], device=boxes.device)
            loss[name_giou] = torch.as_tensor([0.], device=boxes.device)
            return loss

        # prepare positive samples
        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)

        # Compute L1 loss
        loss[name_bbox] = F.l1_loss(src_bbox, target_bbox, reduction='none')
        loss[name_bbox] = loss[name_bbox].sum() / num_gts
        loss[name_bbox] = self.loss_coeff['bbox'] * loss[name_bbox]
        
        # Compute GIoU loss
        loss[name_giou] = self.giou_loss(box_cxcywh_to_xyxy(src_bbox),
                                         box_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]

        return loss

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix=""):
        loss_class = []
        loss_bbox, loss_giou = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            if dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    )
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(box_cxcywh_to_xyxy(src_bbox),
                                         box_cxcywh_to_xyxy(target_bbox))
                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])

        loss = {
            "loss_class_aux" + postfix: sum(loss_class),
            "loss_bbox_aux"  + postfix: sum(loss_bbox),
            "loss_giou_aux"  + postfix: sum(loss_giou)
        }

        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = torch.cat([
            torch.gather(t, 0, dst.to(t.device)) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = torch.cat([t[I] if len(I) > 0 else torch.zeros([0, t.shape[-1]], device=src.device)
            for t, (I, _) in zip(src, match_indices)
        ])

        target_assign = torch.cat([t[J] if len(J) > 0 else torch.zeros([0, t.shape[-1]], device=src.device)
            for t, (_, J) in zip(target, match_indices)
        ])

        return src_assign, target_assign

    def _get_num_gts(self, targets):
        num_gts = sum(len(a) for a in targets)
        num_gts = torch.as_tensor([num_gts], device=targets[0].device).float()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_gts)
        num_gts = torch.clamp(num_gts / get_world_size(), min=1).item()

        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             gt_bbox,
                             gt_class,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(boxes, logits, gt_bbox, gt_class)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(box_cxcywh_to_xyxy(src_bbox),
                                     box_cxcywh_to_xyxy(target_bbox))
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
                                postfix))

        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    ))

        return total_loss

class DINOLoss(DETRLoss):
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(DINOLoss, self).forward(
            boxes, logits, gt_bbox, gt_class, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': torch.as_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt).long()
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype="int64"),
                                         torch.zeros([0], dtype="int64")))
        return dn_match_indices
