# Real-time Convolutional Object Detector

# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .rtcdet_backbone import build_backbone
from .rtcdet_neck import build_neck
from .rtcdet_pafpn import build_fpn
from .rtcdet_head import build_det_head, build_seg_head, build_pose_head
from .rtcdet_pred import build_det_pred, build_seg_pred, build_pose_pred

# --------------- External components ---------------
from utils.misc import multiclass_nms


# Real-time Convolutional General Object Detector
class RTCDet(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes = 20,
                 conf_thresh = 0.01,
                 nms_thresh  = 0.5,
                 topk        = 1000,
                 trainable   = False,
                 deploy      = False,
                 no_multi_labels = False,
                 nms_class_agnostic = False,
                 ):
        super(RTCDet, self).__init__()
        # ---------------- Basic settings ----------------
        ## Basic parameters
        self.cfg = cfg
        self.device = device
        self.deploy = deploy
        self.trainable = trainable
        self.num_classes = num_classes
        ## Network parameters
        self.strides = cfg['stride']
        self.reg_max = cfg['det_head']['reg_max']
        self.num_levels = len(self.strides)
        ## Post-process parameters
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.topk_candidates = topk
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic
        
        # ---------------- Network settings ----------------
        ## Backbone
        self.backbone, self.fpn_feat_dims = build_backbone(cfg, pretrained=cfg['bk_pretrained']&trainable)

        ## Neck: SPP
        self.neck = build_neck(cfg, self.fpn_feat_dims[-1], self.fpn_feat_dims[-1])
        self.fpn_feat_dims[-1] = self.neck.out_dim
        
        ## Neck: FPN
        self.fpn = build_fpn(cfg, self.fpn_feat_dims)
        self.fpn_dims = self.fpn.out_dim
        self.cls_head_dim = max(self.fpn_dims[0], min(num_classes, 100))
        self.reg_head_dim = max(self.fpn_dims[0]//4, 16, 4*self.reg_max)

        ## Head
        self.det_head = nn.Sequential(
            build_det_head(cfg['det_head'], self.fpn_dims, self.cls_head_dim, self.reg_head_dim, self.num_levels),
            build_det_pred(self.cls_head_dim, self.reg_head_dim, self.strides, num_classes, 4, self.reg_max, self.num_levels)
        )
        self.seg_head = nn.Sequential(
            build_seg_head(cfg['seg_head']),
            build_seg_pred()
        ) if cfg['seg_head']['name'] is not None else None
        self.pos_head = nn.Sequential(
            build_pose_head(cfg['pos_head']),
            build_pose_pred()
        ) if cfg['pos_head']['name'] is not None else None

    # Post process
    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List[np.array] -> [[M, C], ...]
            box_preds: List[np.array] -> [[M, 4], ...]
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """
        assert len(cls_preds) == self.num_levels
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]
            if self.no_multi_labels:
                # [M,]
                scores, labels = torch.max(cls_pred_i.sigmoid(), dim=1)

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

                # topk candidates
                predicted_prob, topk_idxs = scores.sort(descending=True)
                topk_scores = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                # filter out the proposals with low confidence score
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

                labels = labels[topk_idxs]
                bboxes = box_pred_i[topk_idxs]
            else:
                # [M, C] -> [MC,]
                scores_i = cls_pred_i.sigmoid().flatten()

                # Keep top k top scoring indices only.
                num_topk = min(self.topk_candidates, box_pred_i.size(0))

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

        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        bboxes = torch.cat(all_bboxes, dim=0)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return bboxes, scores, labels
    
    # Main process
    def forward(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck: SPP ----------------
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # ---------------- Neck: PaFPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Head ----------------
        det_outpus = self.forward_det_head(pyramid_feats)
        seg_outpus = self.forward_seg_head(pyramid_feats)
        pos_outpus = self.forward_pos_head(pyramid_feats)
        outputs = {
            'det_outputs': det_outpus,
            'seg_outputs': seg_outpus,
            'pos_outputs': pos_outpus
        }

        if not self.trainable:
            if seg_outpus is not None:
                det_outpus.update(seg_outpus)
            if pos_outpus is not None:
                det_outpus.update(pos_outpus)
            outputs = det_outpus
        
        else:
            outputs = {
                'det_outputs': det_outpus,
                'seg_outputs': seg_outpus,
                'pos_outputs': pos_outpus
            }

        return outputs

    def forward_det_head(self, x):
        # ---------------- Heads ----------------
        outputs = self.det_head(x)

        # ---------------- Post-process ----------------
        if not self.trainable:
            all_cls_preds = outputs['pred_cls']
            all_box_preds = outputs['pred_box']

            if self.deploy:
                cls_preds = torch.cat(all_cls_preds, dim=1)[0]
                box_preds = torch.cat(all_box_preds, dim=1)[0]
                scores = cls_preds.sigmoid()
                bboxes = box_preds
                # [n_anchors_all, 4 + C]
                outputs = torch.cat([bboxes, scores], dim=-1)

            else:
                # post process
                bboxes, scores, labels = self.post_process(all_cls_preds, all_box_preds)

                outputs = {
                    "scores": scores,
                    "labels": labels,
                    "bboxes": bboxes
                }
            
        return outputs

    def forward_seg_head(self, x):
        if self.seg_head is None:
            return None
    
    def forward_pos_head(self, x):
        if self.pos_head is None:
            return None
