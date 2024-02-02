import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .loss_utils import sigmoid_focal_loss
    from .loss_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou, bbox2delta
    from .loss_utils import is_dist_avail_and_initialized, get_world_size
    from .matcher import HungarianMatcher
except:
    from loss_utils import sigmoid_focal_loss
    from loss_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou, bbox2delta
    from loss_utils import is_dist_avail_and_initialized, get_world_size
    from matcher import HungarianMatcher


class Criterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes=80, aux_loss=False):
        super().__init__()
        # ------------ Basic parameters ------------
        self.cfg = cfg
        self.num_classes = num_classes
        self.k_one2many = cfg['k_one2many']
        self.lambda_one2many = cfg['lambda_one2many']
        self.aux_loss = aux_loss
        self.losses = ['labels', 'boxes']
        # ------------- Focal loss -------------
        self.alpha = 0.25
        self.gamma = 2.0
        # ------------ Matcher ------------
        self.matcher = HungarianMatcher(cost_class = cfg['matcher_hpy']['cost_class'],
                                        cost_bbox  = cfg['matcher_hpy']['cost_bbox'],
                                        cost_giou  = cfg['matcher_hpy']['cost_giou']
                                        )
        # ------------- Loss weight -------------
        self.weight_dict = {'loss_cls':  cfg['loss_coeff']['class'],
                            'loss_box':  cfg['loss_coeff']['bbox'],
                            'loss_giou': cfg['loss_coeff']['giou']}
        if aux_loss:
            aux_weight_dict = {}
            for i in range(cfg['de_num_layers'] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        # ------------- One2many loss weight -------------
        if cfg['num_queries_one2many'] > 0:
            one2many_loss_weight = {}
            for k, v in self.weight_dict.items():
                one2many_loss_weight[k+"_one2many"] = v
            self.weight_dict.update(one2many_loss_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # prepare class targets
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        # to one-hot labels
        target_classes_onehot = torch.zeros([*src_logits.shape[:2], self.num_classes + 1],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]

        # focal loss
        loss_cls = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, self.alpha, self.gamma)

        losses = {}
        losses['loss_cls'] = loss_cls * src_logits.shape[1]

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # prepare bbox targets
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_boxes.device)
        
        # compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, box_xyxy_to_cxcywh(target_boxes), reduction='none')
        src_deltas = outputs["pred_deltas"][idx]
        src_boxes_old = outputs["pred_boxes_old"][idx]
        target_deltas = bbox2delta(src_boxes_old, target_boxes)
        loss_bbox = F.l1_loss(src_deltas, target_deltas, reduction="none")

        # compute GIoU loss
        bbox_giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                        box_cxcywh_to_xyxy(target_boxes))
        loss_giou = 1 - torch.diag(bbox_giou)
        
        losses = {}
        losses['loss_box'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def compute_loss(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def forward(self, outputs, targets):
        # --------------------- One-to-one losses ---------------------
        outputs_one2one = {k: v for k, v in outputs.items() if "one2many" not in k}
        loss_dict = self.compute_loss(outputs_one2one, targets)

        # --------------------- One-to-many losses ---------------------
        outputs_one2many = {k[:-9]: v for k, v in outputs.items() if "one2many" in k}
        if len(outputs_one2many) > 0:
            # Copy targets
            multi_targets = copy.deepcopy(targets)
            for target in multi_targets:
                target["boxes"] = target["boxes"].repeat(self.k_one2many, 1)
                target["labels"] = target["labels"].repeat(self.k_one2many)
            # Compute one-to-many losses
            one2many_loss_dict = self.compute_loss(outputs_one2many, multi_targets)
            # add one2many losses in to the final loss_dict
            for k, v in one2many_loss_dict.items():
                if k + "_one2many" in loss_dict.keys():
                    loss_dict[k + "_one2many"] += v * self.lambda_one2many
                else:
                    loss_dict[k + "_one2many"] = v * self.lambda_one2many

        return loss_dict
    
# build criterion
def build_criterion(cfg, num_classes, aux_loss=True):
    criterion = Criterion(cfg, num_classes, aux_loss)

    return criterion
    