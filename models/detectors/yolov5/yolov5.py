import numpy as np

import torch
import torch.nn as nn

from .yolov5_backbone import build_backbone
from .yolov5_pafpn import build_fpn
from .yolov5_head import build_head

from utils.misc import multiclass_nms


class YOLOv5(nn.Module):
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
                 nms_class_agnostic = False):
        super(YOLOv5, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk_candidates = topk
        self.no_multi_labels = no_multi_labels
        self.nms_class_agnostic = nms_class_agnostic
        self.deploy = deploy
        
        # ------------------- Anchor box -------------------
        self.num_levels = 3
        self.num_anchors = len(cfg['anchor_size']) // self.num_levels
        self.anchor_size = torch.as_tensor(
            cfg['anchor_size']
            ).float().view(self.num_levels, self.num_anchors, 2) # [S, A, 2]
        
        # ------------------- Network Structure -------------------
        ## Backbone
        self.backbone, feats_dim = build_backbone(cfg)
        
        ## FPN
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim, out_dim=round(256*cfg['width']))
        self.head_dim = self.fpn.out_dim

        ## Head
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, head_dim, head_dim, num_classes) 
            for head_dim in self.head_dim
            ])

        ## Pred
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 1 * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4 * self.num_anchors, kernel_size=1) 
                                for head in self.non_shared_heads
                              ])                 


    # ---------------------- Basic Functions ----------------------
    ## generate anchor points
    def generate_anchors(self, level, fmp_size):
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size[level]

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
        
    ## post-process
    def post_process(self, cls_preds, box_preds, obj_preds=None):
        """
        Input:
            cls_preds: List[np.array] -> [[M, C], ...]
            box_preds: List[np.array] -> [[M, 4], ...]
            obj_preds: List[np.array] -> [[M, 1], ...] or None
        Output:
            bboxes: np.array -> [N, 4]
            scores: np.array -> [N,]
            labels: np.array -> [N,]
        """
        assert len(cls_preds) == self.num_levels
        all_scores = []
        all_labels = []
        all_bboxes = []

        for level in range(self.num_levels):
            cls_pred_i = cls_preds[level]
            box_pred_i = box_preds[level]
            num_topk = min(self.topk_candidates, box_pred_i.shape[0])

            # filter out by objectness
            obj_preds_i = obj_preds[level]
            keep_idxs = obj_preds_i[..., 0] > self.conf_thresh
            cls_pred_i = obj_preds_i[keep_idxs] * cls_pred_i[keep_idxs]
            box_pred_i = box_pred_i[keep_idxs]

            if self.no_multi_labels:
                # [M,]
                scores_i, labels_i = np.max(cls_pred_i, axis=1), np.argmax(cls_pred_i, axis=1)

                # topk candidates
                topk_idxs = np.argsort(-scores_i)
                topk_idxs = topk_idxs[:num_topk]
                scores_i = scores_i[topk_idxs]
                labels_i = labels_i[topk_idxs]
                bboxes_i = box_pred_i[topk_idxs]
            else:
                # [M, C] -> [MC,]
                scores_i = cls_pred_i.flatten()

                # topk candidates
                predicted_prob, topk_idxs = np.sort(scores_i)[::-1], np.argsort(-scores_i)
                scores_i = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[:num_topk]

                anchor_idxs = topk_idxs // self.num_classes
                bboxes_i = box_pred_i[anchor_idxs]

                labels_i = topk_idxs % self.num_classes

            all_scores.append(scores_i)
            all_labels.append(labels_i)
            all_bboxes.append(bboxes_i)

        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        bboxes = np.concatenate(all_bboxes, axis=0)

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return bboxes, scores, labels

    # ---------------------- Main Process for Inference ----------------------
    @torch.no_grad()
    def inference_single_image(self, x):
        # backbone
        pyramid_feats = self.backbone(x)

        # fpn
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_anchors = []
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # anchors: [M, 4]
            fmp_size = cls_pred.shape[-2:]
            anchors = self.generate_anchors(level, fmp_size)

            # [1, C, H, W] -> [H, W, C] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            # decode bbox
            ctr_pred = (torch.sigmoid(reg_pred[..., :2]) * 2.0 - 0.5 + anchors[..., :2]) * self.stride[level]
            wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
            pred_x1y1 = ctr_pred - wh_pred * 0.5
            pred_x2y2 = ctr_pred + wh_pred * 0.5
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        if self.deploy:
            obj_preds = torch.cat(all_obj_preds, dim=0)
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_preds = torch.cat(all_box_preds, dim=0)
            scores = torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid())
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            # post process
            obj_preds = [obj_pred_i.sigmoid().cpu().numpy() for obj_pred_i in all_obj_preds]
            cls_preds = [cls_pred_i.sigmoid().cpu().numpy() for cls_pred_i in all_cls_preds]
            box_preds = [box_pred_i.cpu().numpy()           for box_pred_i in all_box_preds]
            bboxes, scores, labels = self.post_process(cls_preds, box_preds, obj_preds)
        
            return bboxes, scores, labels


    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            pyramid_feats = self.backbone(x)

            # fpn
            pyramid_feats = self.fpn(pyramid_feats)

            # non-shared heads
            all_fmp_sizes = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # [B, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

                # decode bbox
                ctr_pred = (torch.sigmoid(reg_pred[..., :2]) * 2.0 - 0.5 + anchors[..., :2]) * self.stride[level]
                wh_pred = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]
                pred_x1y1 = ctr_pred - wh_pred * 0.5
                pred_x2y2 = ctr_pred + wh_pred * 0.5
                box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_fmp_sizes.append(fmp_size)
            
            # output dict
            outputs = {"pred_obj": all_obj_preds,        # List [B, M, 1]
                       "pred_cls": all_cls_preds,        # List [B, M, C]
                       "pred_box": all_box_preds,        # List [B, M, 4]
                       'fmp_sizes': all_fmp_sizes,       # List
                       'strides': self.stride,           # List
                       }

            return outputs 
