# Objects as Points

# --------------- Torch components ---------------
import torch
import torch.nn as nn

# --------------- Model components ---------------
from .ctrnet_encoder import build_encoder
from .ctrnet_decoder import build_decoder
from .ctrnet_neck    import build_neck
from .ctrnet_head    import build_det_head
from .ctrnet_pred    import build_det_pred


# CenterNet
class CenterNet(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes = 20,
                 conf_thresh = 0.01,
                 topk        = 1000,
                 trainable   = False,
                 deploy      = False,
                 no_multi_labels = False,
                 nms_class_agnostic = False,
                 ):
        super(CenterNet, self).__init__()
        # ---------------- Basic Parameters ----------------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['out_stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes
        self.topk_candidates = topk
        self.deploy = deploy
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic
        self.head_dim = round(256 * cfg['width'])
        
        # ---------------- Network Parameters ----------------
        ## Encoder
        self.encoder, feat_dims = build_encoder(cfg)

        ## Neck
        self.neck = build_neck(cfg, feat_dims[-1], feat_dims[-1])
        self.feat_dim = self.neck.out_dim
        
        ## Decoder
        self.decoder = build_decoder(cfg, self.feat_dim, self.head_dim)

        ## Head
        self.det_head = nn.Sequential(
            build_det_head(cfg, self.head_dim, self.head_dim),
            build_det_pred(self.head_dim, self.head_dim, self.stride, num_classes, 4)
        )
        ## Aux Head
        self.aux_det_head = nn.Sequential(
            build_det_head(cfg, self.head_dim, self.head_dim),
            build_det_pred(self.head_dim, self.head_dim, self.stride, num_classes, 4)
        )

    # Post process
    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: torch.Tensor -> [M, C]
            box_pred: torch.Tensor -> [M, 4]
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]
        if self.no_multi_labels:
            # [M,]
            scores, labels = torch.max(cls_pred.sigmoid(), dim=1)

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_pred.size(0))

            # topk candidates
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            labels = labels[topk_idxs]
            bboxes = box_pred[topk_idxs]
        else:
            # [M, C] -> [MC,]
            scores = cls_pred.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_pred.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_pred[anchor_idxs]

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        return bboxes, scores, labels
    
    # Main process
    def forward(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.encoder(x)

        # ---------------- Neck ----------------
        feat = self.neck(pyramid_feats[-1])

        # ---------------- Encoder ----------------
        feat = self.decoder(feat)

        # ---------------- Head ----------------
        outputs = self.det_head(feat)
        if self.trainable:
            outputs['aux_outputs'] = self.aux_det_head(feat)

        # ---------------- Post-process ----------------
        if not self.trainable:
            cls_preds = outputs['pred_cls']
            box_preds = outputs['pred_box']

            if self.deploy:
                scores = cls_preds[0].sigmoid()
                bboxes = box_preds[0]
                # [n_anchors_all, 4 + C]
                outputs = torch.cat([bboxes, scores], dim=-1)

            else:
                # post process
                bboxes, scores, labels = self.post_process(cls_preds, box_preds)

                outputs = {
                    "scores": scores,
                    "labels": labels,
                    "bboxes": bboxes
                }
            
        return outputs
    