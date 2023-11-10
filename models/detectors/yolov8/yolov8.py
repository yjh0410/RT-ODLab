# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .yolov8_backbone import build_backbone
from .yolov8_neck import build_neck
from .yolov8_pafpn import build_fpn
from .yolov8_head import build_det_head
from .yolov8_pred import build_pred_layer

# --------------- External components ---------------
from utils.misc import multiclass_nms


# YOLOv8
class YOLOv8(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes = 20,
                 conf_thresh = 0.01,
                 nms_thresh  = 0.5,
                 topk        = 1000,
                 trainable   = False,
                 deploy      = False,
                 nms_class_agnostic = False):
        super(YOLOv8, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.strides = cfg['stride']
        self.reg_max = cfg['reg_max']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_levels = len(self.strides)
        self.num_classes = num_classes
        self.topk = topk
        self.deploy = deploy
        self.nms_class_agnostic = nms_class_agnostic
        
        # ---------------------- Network Parameters ----------------------
        ## ----------- Backbone -----------
        self.backbone, feat_dims = build_backbone(cfg)

        ## ----------- Neck: SPP -----------
        self.neck = build_neck(cfg, feat_dims[-1], feat_dims[-1])
        feat_dims[-1] = self.neck.out_dim
        
        ## ----------- Neck: FPN -----------
        self.fpn = build_fpn(cfg, feat_dims)
        self.fpn_dims = self.fpn.out_dim

        ## ----------- Heads -----------
        self.det_heads = build_det_head(cfg, self.fpn_dims, self.num_levels, num_classes, self.reg_max)

        ## ----------- Preds -----------
        self.pred_layers = build_pred_layer(cls_dim     = self.det_heads.cls_head_dim,
                                            reg_dim     = self.det_heads.reg_head_dim,
                                            strides     = self.strides,
                                            num_classes = num_classes,
                                            num_coords  = 4,
                                            num_levels  = self.num_levels,
                                            reg_max     = self.reg_max)

    ## post-process
    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[H x W, C], ...]
            box_preds: List(Tensor) [[H x W, 4], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            
            # (H x W x C,)
            scores_i = cls_pred_i.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, box_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)
        
        return bboxes, scores, labels

    # ---------------------- Main Process for Inference ----------------------
    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck: SPP ----------------
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # ---------------- Neck: PaFPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        cls_feats, reg_feats = self.det_heads(pyramid_feats)

        # ---------------- Preds ----------------
        outputs = self.pred_layers(cls_feats, reg_feats)

        all_cls_preds = outputs['pred_cls']
        all_box_preds = outputs['pred_box']

        if self.deploy:
            cls_preds = torch.cat(all_cls_preds, dim=1)[0]
            box_preds = torch.cat(all_box_preds, dim=1)[0]
            scores = cls_preds.sigmoid()
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            # post process
            bboxes, scores, labels = self.post_process(all_cls_preds, all_box_preds)
        
            return bboxes, scores, labels

    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)

            # ---------------- Neck: SPP ----------------
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # ---------------- Neck: PaFPN ----------------
            pyramid_feats = self.fpn(pyramid_feats)

            # ---------------- Heads ----------------
            cls_feats, reg_feats = self.det_heads(pyramid_feats)

            # ---------------- Preds ----------------
            outputs = self.pred_layers(cls_feats, reg_feats)
            
            return outputs 
