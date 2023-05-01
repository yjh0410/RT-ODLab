import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov8_backbone import build_backbone
from .yolov8_neck import build_neck
from .yolov8_pafpn import build_fpn
from .yolov8_head import build_head

from utils.misc import multiclass_nms


# Anchor-free YOLO
class YOLOv8(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLOv8, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.reg_max = cfg['reg_max']
        self.use_dfl = cfg['reg_max'] > 1
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        
        # --------- Network Parameters ----------
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)

        ## backbone
        self.backbone, feats_dim = build_backbone(cfg, cfg['pretrained']*trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim)
        fpn_dims = self.fpn.out_dim

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg, feat_dim, fpn_dims, num_classes) 
            for feat_dim in fpn_dims
            ])

        ## pred
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head.cls_out_dim, self.num_classes, kernel_size=1) 
                                for head in self.non_shared_heads
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head.reg_out_dim, 4*(cfg['reg_max']), kernel_size=1) 
                                for head in self.non_shared_heads
                              ])                 


    # ---------------------- Basic Functions ----------------------
    ## generate anchor points
    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        
    ## post-process
    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[H x W, C], ...]
            box_preds: List(Tensor) [[H x W, 4], ...]
            anchors:   List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            # (H x W x KA x C,)
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
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels


    # ---------------------- Main Process for Inference ----------------------
    @torch.no_grad()
    def inference_single_image(self, x):
        # backbone
        pyramid_feats = self.backbone(x)

        # neck
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # fpn
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_cls_preds = []
        all_box_preds = []
        all_anchors = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # pred
            cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
            reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max), H, W]

            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [M, 2]
            anchors = self.generate_anchors(level, fmp_size)

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

            # decode bbox
            if self.use_dfl:
                B, M = reg_pred.shape[:2]
                # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
                reg_pred = reg_pred.reshape([B, M, 4, self.reg_max])
                # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
                reg_pred = reg_pred.permute(0, 3, 2, 1).contiguous()
                # [B, reg_max, 4, M] -> [B, 1, 4, M]
                reg_pred = self.proj_conv(F.softmax(reg_pred, dim=1))
                # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
                reg_pred = reg_pred.view(B, 4, M).permute(0, 2, 1).contiguous()
            pred_x1y1 = anchors - reg_pred[..., :2] * self.stride[level]
            pred_x2y2 = anchors + reg_pred[..., 2:] * self.stride[level]
            box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        # post process
        bboxes, scores, labels = self.post_process(
            all_cls_preds, all_box_preds, all_anchors)
        
        return bboxes, scores, labels


    # ---------------------- Main Process for Training ----------------------
    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            pyramid_feats = self.backbone(x)

            # neck
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # fpn
            pyramid_feats = self.fpn(pyramid_feats)

            # non-shared heads
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_box_preds = []
            all_strides = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # pred
                cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
                reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max), H, W]

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 2]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

                # decode bbox
                if self.use_dfl:
                    B, M = reg_pred.shape[:2]
                    # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
                    reg_pred_ = reg_pred.reshape([B, M, 4, self.reg_max]).clone()
                    # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
                    reg_pred_ = reg_pred_.permute(0, 3, 2, 1).contiguous()
                    # [B, reg_max, 4, M] -> [B, 1, 4, M]
                    reg_pred_ = self.proj_conv(F.softmax(reg_pred_, dim=1))
                    # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
                    reg_pred_ = reg_pred_.view(B, 4, M).permute(0, 2, 1).contiguous()
                pred_x1y1 = anchors - reg_pred_[..., :2] * self.stride[level]
                pred_x2y2 = anchors + reg_pred_[..., 2:] * self.stride[level]
                box_pred = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

                del reg_pred_

                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)
            
            # output dict
            outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4*(reg_max)]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [M, 2]
                       "strides": self.stride,           # List(Int) = [8, 16, 32]
                       "stride_tensor": all_strides      # List(Tensor) [M, 1]
                       }           
            return outputs